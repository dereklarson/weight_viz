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
export function buildNetwork(networkShape: number[], inputIds: string[]): Node[][] {
  let numLayers = networkShape.length;
  /** List of layers, with each layer being a list of nodes. */
  let network: Node[][] = [];
  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    let isOutputLayer = layerIdx === numLayers - 1;
    let isInputLayer = layerIdx === 0;
    let currentLayer: Node[] = [];
    network.push(currentLayer);
    let numNodes = networkShape[layerIdx];
    for (let i = 0; i < numNodes; i++) {
      let nodeId = "";
      if (isInputLayer) {
        nodeId = inputIds[i];
      } else {
        nodeId = `${layerIdx}_${i}`
      }
      let node = new Node(nodeId);
      currentLayer.push(node);
      if (layerIdx >= 1) {
        // Add links from nodes in the previous layer to this node.
        for (let j = 0; j < network[layerIdx - 1].length; j++) {
          let prevNode = network[layerIdx - 1][j];
          let link = new Link(prevNode, node);
          prevNode.outputs.push(link);
          node.inputLinks.push(link);
        }
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
export function updateWeights(network: Node[][], frame_heads: number[][][], selectedTokenId: number) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let cols = new Array(10).fill(0);
      if (selectedTokenId) {
        cols = frame_heads[i][selectedTokenId]
      }
      else {
        cols = normColSum(frame_heads[i])
      }
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

/** Returns the output node in the network. */
export function getOutputNode(network: Node[][]) {
  return network[network.length - 1][0];
}
