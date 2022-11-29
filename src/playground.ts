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

import * as d3 from 'd3';
import { HeatMap } from "./heatmap";
import { AppendingLineChart } from "./linechart";
import * as nn from "./nn";
import {
  experiments, getKeyFromValue, problems, State
} from "./state";

let mainWidth;

const TOKEN_SIZE = 20;
const RECT_SIZE = 80;

enum HoverType {
  BIAS, WEIGHT
}

// TODO Add an interface to support more complicated tokens
// interface InputFeature {
//   label?: string;
// }

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      step(1);
      return false;  // Not done.
    }, 0);
  }
}

let state = State.deserializeState();

let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
  new HeatMap(300, 10, xDomain, xDomain, d3.select("#heatmap"),
    { showAxes: true });
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);
let network: nn.Node[][] = null;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
  ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    state.currentFrame = 0;
    reset();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    step(1);
  });

  d3.select("#prev-step-button").on("click", () => {
    player.pause();
    step(-1);
  });


  d3.select("#load-experiment-button").on("click", () => {
    loadData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function () {
    let newExperiment = experiments[this.dataset.dataset];
    if (newExperiment === state.experiment) {
      return; // No-op.
    }
    state.experiment = newExperiment;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    loadData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(experiments, state.experiment);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let batchSize = d3.select("#batchSize").on("input", function () {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let problem = d3.select("#problem").on("change", function () {
    state.problem = problems[this.value];
    loadData();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.

  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
      .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
    }
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
          .style({
            "stroke-dashoffset": -state.currentFrame / 3,
            "stroke-width": linkWidthScale(Math.abs(link.weight)),
            "stroke": colorScale(link.weight)
          })
          .datum(link);
      }
    }
  }
}

function drawInputNode(cx: number, cy: number, nodeId: string,
  container, node?: nn.Node) {
  let dimension = TOKEN_SIZE
  let x = cx - dimension / 2;
  let y = cy - dimension / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  let rect = nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: dimension,
      height: dimension,
    })
    .on("mouseenter", function () {
      selectedNodeId = nodeId;
      rect.classed("hovered", true);
      nodeGroup.classed("hovered", true);
    })
    .on("mouseleave", function () {
      selectedNodeId = null;
      rect.classed("hovered", false);
      nodeGroup.classed("hovered", false);
    })
    .on("click", function () {
      state.nodeState[nodeId] = !state.nodeState[nodeId];
      parametersChanged = true;
      reset()
    });

  let activeOrNotClass = state.nodeState[nodeId] ? "active" : "inactive";
  let text = nodeGroup.append("text").attr({
    class: "main-label",
    x: dimension / 2,
    y: dimension / 2, "text-anchor": "end"
  });
  text.append("tspan").text(nodeId)
  nodeGroup.classed(activeOrNotClass, true);
}


function drawNode(cx: number, cy: number, nodeId: string, container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    });

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("mouseenter", function () {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      heatMap.updateBackground(state.frames[state.currentFrame]["heads"][parseInt(nodeId) % 4]);
    })
    .on("mouseleave", function () {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      // heatMap.updateBackground(boundary[nn.getOutputNode(network).id]);
    });
  let nodeHeatMap = new HeatMap(RECT_SIZE, 10, xDomain,
    xDomain, div, { noSvg: true });
  div.datum({ heatmap: nodeHeatMap, id: nodeId });

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: { [id: string]: { cx: number, cy: number } } = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
    .domain(d3.range(1, numLayers - 1))
    .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number, dim: number) => nodeIndex * (dim + 25);


  // Draw the input layer.
  let cx = TOKEN_SIZE / 2 + 50;
  let maxY = nodeIndexScale(state.inputIds.length, TOKEN_SIZE);
  state.inputIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i, TOKEN_SIZE) + TOKEN_SIZE / 2;
    node2coord[nodeId] = { cx, cy };
    drawInputNode(cx, cy, nodeId, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes, RECT_SIZE));
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i, RECT_SIZE) + RECT_SIZE / 2;
      node2coord[node.id] = { cx, cy };
      drawNode(cx, cy, node.id, container, node);

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        drawLink(link, node2coord, network,
          container, j === 0, j, node.inputLinks.length).node() as any;
      }
    }
  }

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0, RECT_SIZE) + RECT_SIZE / 2;
  node2coord[node.id] = { cx, cy };
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
      node.inputLinks.length);
  }
  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = getRelativeHeight(d3.select("#network"))
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
  coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function () {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI();
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
  input: nn.Link, node2coord: { [id: string]: { cx: number, cy: number } },
  network: nn.Node[][], container,
  isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let dimension = RECT_SIZE;
  if (state.inputIds.includes(input.source.id)) {
    dimension = TOKEN_SIZE;
  }
  let datum = {
    source: {
      y: source.cx + dimension / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function () {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function () {
      updateHoverCard(null);
    });
  return line;
}


// function updateAttentionPattern(network: nn.Node[][]) {
//   let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
//   let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);
//   let i = 0, j = 0;
// }

function updateUI(firstStep = false) {
  console.log(state)
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  let selectedId = selectedNodeId != null ?
    selectedNodeId : nn.getOutputNode(network).id;

  d3.select("#network").selectAll("div.canvas")
    .each(function (data: { heatmap: HeatMap, id: string }) {
      try {
        data.heatmap.updateBackground(state.frames[state.currentFrame]["heads"][parseInt(data.id) % 4])
      }
      catch {
        data.heatmap.updateBackground(state.nullFrame["heads"][0])
      }
    });


  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  let frame = state.nullFrame
  if (state.frames) {
    frame = state.frames[state.currentFrame]
  }
  d3.select("#loss-train").text(humanReadable(frame.lossTrain));
  d3.select("#loss-test").text(humanReadable(frame.lossTest));
  d3.select("#epoch-number").text(addCommas(zeroPad(frame.epoch)));
  lineChart.addDataPoint([frame.lossTrain, frame.lossTest]);
}

function step(count: number): void {
  state.currentFrame += count;
  if (state.currentFrame >= 99) {
    state.currentFrame = 99
    player.pause();
  }
  else if (state.currentFrame < 0) {
    state.currentFrame = 0
  }
  updateUI();
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset() {
  lineChart.reset();
  state.serialize();
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  let shape = [state.inputIds.length].concat(state.networkShape).concat([1]);
  network = nn.buildNetwork(shape, state.inputIds, state.initZero);
  console.log(network)
  drawNetwork(network);
  updateUI(true);
};

function loadData() {
  fetch('./test_out.json')
    .then(response => response.json())
    .then(data => {
      console.log(data)
      state.dEmbed = data['d_embed'];
    })
    .catch(error => console.log(error));

  fetch('./test_data.json')
    .then(response => response.json())
    .then(data => {
      state.frames = data;
    })
    .catch(error => console.log(error));
}

let parametersChanged = false;


makeGUI();
loadData();
reset();
