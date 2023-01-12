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

import { add, clone, matrix, multiply, transpose } from "./minMathjs";

// Mathjs Matrix type was difficult to extract via minMathjs
type Matrix = any;

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
  accuracyTest: number,
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
  label: string;
  inLinks: Link[] = [];
  outLinks: Link[] = [];
  position: number[] = [0, 0]
  shape: number[] = [0, 0]

  // Creates a new node with the provided id
  constructor(id: string, label: string = "") {
    this.id = id;
    this.label = label;
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
 * @param blocks A list representing the number of attention heads per
 *   Transformer block in the netowrk. A two-block network with 8 heads each
 *   has network = [8, 8].
 * @param inputs List of ids for the input nodes. This might be a selection
 *   of the vocabulary, or the current context.
 */
export function buildNetwork(blocks: number[], inputs: string[], maxHeads: number): Node[][] {
  let network: Node[][] = [inputs.map((name, idx) => new Node(String(idx), name))];
  for (let blockIdx = 0; blockIdx < blocks.length; blockIdx++) {
    let currentBlock: Node[] = [];
    network.push(currentBlock);
    let numNodes = Math.min(blocks[blockIdx], maxHeads);
    for (let i = 0; i < numNodes; i++) {
      let node = new Node(`${blockIdx}_${i}`)
      currentBlock.push(node);
      // Add links from nodes in the previous layer to this node.
      for (let j = 0; j < network[blockIdx].length; j++) {
        let prevNode = network[blockIdx][j];
        let link = new Link(prevNode, node);
        prevNode.outLinks.push(link);
        node.inLinks.push(link);
      }
    }
  }
  return network;
}

function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const scores = logits.map(l => Math.exp(l - maxLogit));
  const denom = scores.reduce((acc, cur) => acc + cur);
  return scores.map(s => s / denom);
}

export function maskAndScale(attn: Array2D | Matrix, scale: number) {
  return matrix(attn).map((val, index) => index[0] >= index[1] ? val / scale : -1e9)
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
  // Calculate Argmax of the last row of the final residual
  let resRow = unembed.toArray()[unembed.size()[0] - 1]
  let result = resRow.indexOf(Math.max(...resRow))

  return { position, embedding, preBlock, block1, unembed, attn_weights, result }
}

function aggProbs(qkMatrix: Array2D): number[] {
  var sum = (base, acc) => base.map((val, idx) => acc[idx] + val)
  var weights = qkMatrix.reduce(sum)
  return softmax(weights)
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
      let probs = selectedTokenId !== null ? softmax(frame_heads[i][parseInt(selectedTokenId)]) : aggProbs(frame_heads[i])
      if (blockIdx - 1 === inspBlockIdx && inspHeadIdx !== i) probs = new Array(probs.length).fill(0)
      // Update the weights coming into this node.
      for (let j = 0; j < node.inLinks.length; j++) {
        let link = node.inLinks[j];
        link.weight = probs[j]
        if (link.isDead) {
          continue;
        }
      }
    }
  }
}

export function parseNodeId(nodeId: string): number[] {
  if (nodeId === null) return [null, null]
  return nodeId.split("_").map(id => parseInt(id))
}