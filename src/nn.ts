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
export interface TransformerConfig {
  n_ctx: number,
  d_embed: number,
  n_heads: number,
  n_blocks: number,
  n_vocab: number,
  vocabulary: string[],
}

type Matrix = number[][];

export interface TransformerBlock {
  attention: Matrix[],
  output: Matrix[],
  mlp: Matrix,
}

export interface Frame {
  epoch: 0,
  lossTest: 1,
  lossTrain: 1,
  embedding: Matrix,
  pos_embed: Matrix,
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
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = blocks[blockIdx];
    for (let i = 0; i < numNodes; i++) {
      let node = new Node(`${blockIdx}_${i}`)
      currentLayer.push(node);
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

function normColSum(matrix: number[][]) {
  var sum = (base, acc) => base.map((val, idx) => acc[idx] + val)
  var cols = matrix.reduce(sum)
  var norm = Math.max(...cols)
  return cols.map(val => val / norm)
}

/**
 * Updates the weights of the network
 */
export function updateWeights(network: Node[][], frame_heads: number[][][], selectedTokenId: string) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let cols = selectedTokenId !== null ? frame_heads[i][parseInt(selectedTokenId)] : normColSum(frame_heads[i])
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
  for (let layerIdx = ignoreInputs ? 1 : 0;
    layerIdx < network.length;
    layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      accessor(node);
    }
  }
}

export function parseNodeId(nodeId: string) {
  return [parseInt(nodeId.split("_")[0]), parseInt(nodeId.split("_")[1])]
}

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
