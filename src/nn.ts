/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import { add, clone, matrix, Matrix, multiply, transpose } from "mathjs";

type Array2D = number[][]

export interface TransformerConfig {
  n_ctx: number,
  d_embed: number,
  d_head: number,
  n_heads: number,
  n_blocks: number,
  n_vocab: number,
  vocabulary: string[],
}

export interface TransformerBlock {
  attention: Array2D[],
  output: Array2D[],
  qk: Array2D[],
  ov: Array2D[],
  mlp: Array2D,
}

export interface WFrame {
  epoch: number,
  lossTest: number,
  lossTrain: number,
  embedding: Array2D,
  unembedding: Array2D,
  pos_embed: Array2D,
  blocks: TransformerBlock[],
}

/**
 * A node in a neural network.
 */
export class Node {
  id: string;
  inputLinks: Link[] = [];
  outputs: Link[] = [];
  totalInput: number;
  output: number;

  // Creates a new node with the provided id
  constructor(id: string) {
    this.id = id;
  }

  /** Recomputes the node's output and returns it. */
  updateOutput(): number {
    // Stores total input into the node.
    for (let j = 0; j < this.inputLinks.length; j++) {
      let link = this.inputLinks[j];
      this.totalInput += link.weight * link.source.output;
    }
    this.output = this.totalInput;
    return this.output;
  }
}

/**
 * A link in a neural network.
 */
export class Link {
  id: string;
  source: Node;
  dest: Node;
  weight: number;
  isDead = false;
  /**
   * Constructs a link in the neural network with initial weight.
   *
   * @param source The source node.
   * @param dest The destination node.
   */
  constructor(source: Node, dest: Node, weight: number = 0) {
    this.id = source.id + "-" + dest.id;
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

/**
 * Builds a neural network.
 *
 * @param networkShape The shape of the network. E.g. [1, 2, 3, 1] means
 *   the network will have one input node, 2 nodes in first hidden layer,
 *   3 nodes in second hidden layer and 1 output node.
 * @param inputIds List of ids for the input nodes.
 */
export function buildNetwork(blocks: number[], vocabulary: string[]): Node[][] {
  // TODO For now, concatenate the 'output' node placeholder
  blocks = blocks.concat(1)
  let numBlocks = blocks.length;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [vocabulary.map(token => new Node(token))];
  for (let blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
    // let isOutputblock = blockIdx === numBlocks - 1;
    let currentBlock: Node[] = [];
    network.push(currentBlock);
    let numNodes = blocks[blockIdx];
    for (let i = 0; i < numNodes; i++) {
      let node = new Node(`${blockIdx}_${i}`)
      currentBlock.push(node);
      // Add links from nodes in the previous layer to this node.
      for (let j = 0; j < network[blockIdx].length; j++) {
        let prevNode = network[blockIdx][j];
        let link = new Link(prevNode, node);
        prevNode.outputs.push(link);
        node.inputLinks.push(link);
      }
    }
  }
  return network;
}

function softmax(logits: number[]) {
  const maxLogit = Math.max(...logits);
  const scores = logits.map(l => Math.exp(l - maxLogit));
  const denom = scores.reduce((acc, cur) => acc + cur);
  return scores.map(s => s / denom);
}

function maskAndScale(attn: Matrix, scale: number) {
  return attn.map((val, index) => index[0] >= index[1] ? val / scale : -1e9)
}

/** Returns the series of residual states propagating through the network */
export function forward(context: number[], frame: WFrame, config: TransformerConfig) {
  var embedding = matrix(context.map((tokenIdx) => frame.embedding[tokenIdx]))
  var position = matrix(frame.pos_embed)
  var preBlock = add(embedding, matrix(frame.pos_embed))
  let block1 = clone(preBlock);
  let attn_weights = []
  for (let blockIdx = 0; blockIdx < frame.blocks.length; blockIdx++) {
    let block = frame.blocks[blockIdx]
    for (let headIdx = 0; headIdx < frame.blocks[0].qk.length; headIdx++) {
      // Calculate weighted attention matrix
      var attn = multiply(multiply(preBlock, transpose(block.qk[headIdx])), transpose(preBlock));
      attn = maskAndScale(attn, Math.sqrt(config.d_head))
      var attn_sm = attn.toArray().map((row) => softmax(row))
      attn_weights.push(attn_sm)
      // Apply attention to OV circuit
      var weighted = multiply(transpose(attn_sm), preBlock)
      var head_sum = multiply(weighted, transpose(block.ov[headIdx]))
      block1 = add(block1, head_sum)
    }
  }
  var unembed = multiply(block1, frame.unembedding)

  return { position, embedding, preBlock, block1, unembed, attn_weights }
}

function normColSum(matrix: Array2D) {
  var sum = (base, acc) => base.map((val, idx) => acc[idx] + val)
  var cols = matrix.reduce(sum)
  var norm = Math.max(...cols)
  return cols.map(val => val / norm)
}

/**
 * Updates the weights of the network
 */
export function updateWeights(network: Node[][], frame_heads: Array2D[],
  selectedTokenId: string, inspectedNodeId: string) {
  var [inspBlockIdx, inspHeadIdx, inspIsOut] = parseNodeId(inspectedNodeId)
  for (let blockIdx = 1; blockIdx < network.length; blockIdx++) {
    let currentBlock = network[blockIdx];
    for (let i = 0; i < currentBlock.length; i++) {
      let node = currentBlock[i];
      let cols = selectedTokenId !== null ? frame_heads[i][parseInt(selectedTokenId)] : normColSum(frame_heads[i])
      if (blockIdx - 1 === inspBlockIdx && inspHeadIdx !== i) cols = Array(cols.length).fill(0)
      // Update the weights coming into this node.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        link.weight = cols[j]
        if (link.isDead) {
          continue;
        }
      }
    }
  }
}

/** Iterates over every node in the network/ */
export function forEachNode(network: Node[][], ignoreInputs: boolean,
  accessor: (node: Node) => any) {
  for (let blockIdx = ignoreInputs ? 1 : 0;
    blockIdx < network.length;
    blockIdx++) {
    let currentBlock = network[blockIdx];
    for (let i = 0; i < currentBlock.length; i++) {
      let node = currentBlock[i];
      accessor(node);
    }
  }
}

export function parseNodeId(nodeId: string) {
  if (nodeId === null) return [null, null]
  return nodeId.split("_").map(id => parseInt(id))
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
