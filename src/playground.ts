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
import { LineChart } from "./linechart";
import * as nn from "./nn";
import { Player } from "./player";
import { State } from "./state";

let mainWidth;

const TOKEN_SIZE = 24;
const RECT_SIZE = 80;

enum HoverType {
  BIAS, WEIGHT
}

// TODO Add an interface to support more complicated tokens
// interface InputFeature {
//   label?: string;
// }


let state = State.deserializeState();

let experiments: string[] = [];
let expTags = { sample: { name: "Sample", tags: ["1"] } }
let currentConfig = {
  vocabulary: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], n_heads: 4, n_blocks: 1
}
let currentFrame = {
  epoch: 0,
  lossTest: 1,
  lossTrain: 1,
  blocks: [
    {
      attention: new Array(4).fill(0).map(_ => new Array(10).fill(0).map(_ => new Array(10).fill(0))),
      output: new Array(4).fill(0).map(_ => new Array(10).fill(0).map(_ => new Array(10).fill(0))),
      mlp: []
    }
  ]
}
let frames = [currentFrame];
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let finalHeatMap =
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
let lineChart = new LineChart(d3.select("#linechart"),
  ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    state.currentFrameIdx = 0;
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
    loadData(expTags[state.experiment].tags[state.seed - 1]);
  });

  let seedSlider = d3.select("#seed").on("input", function () {
    state.seed = this.value;
    d3.select("label[for='seed'] .value").text(this.value);
    // reset();
  });
  seedSlider.property("value", state.seed);
  d3.select("label[for='seed'] .value").text(state.seed);

  let experiment = d3.select("#experiment").on("change", function () {
    state.experiment = this.value;
    state.currentTag = expTags[state.experiment].tags[0];
    d3.select("#seed").property("max", expTags[state.experiment].tags.length)
    loadData(state.currentTag);
    // reset();
  });
  experiment.property("value", state.experiment);

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
            "stroke-dashoffset": -state.currentFrameIdx / 3,
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
    })
    .on("mouseenter", function () {
      selectedNodeId = nodeId;
      rect.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      nn.updateWeights(network, currentFrame.blocks[0].attention, parseInt(nodeId))
      updateUI();
    })
    .on("mouseleave", function () {
      selectedNodeId = null;
      rect.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      nn.updateWeights(network, currentFrame.blocks[0].attention, null)
      updateUI();
    })
    .on("click", function () {
      state.tokenState[nodeId] = !state.tokenState[nodeId];
      reset()
    });

  // Draw the main rectangle.
  let rect = nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: dimension,
      height: dimension,
    })

  let text = nodeGroup.append("text").attr({
    class: "main-label",
    x: dimension / 2,
    y: dimension / 2, "text-anchor": "end"
  });
  text.append("tspan").text(nodeId)

  let activeOrNotClass = state.tokenState[nodeId] ? "active" : "inactive";
  nodeGroup.classed(activeOrNotClass, true);
}


function drawNode(cx: number, cy: number, nodeId: string, container, node?: nn.Node) {
  let x = cx - RECT_SIZE;
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
      width: RECT_SIZE * 2,
      height: RECT_SIZE,
    });

  // Draw the node's canvas.
  let attn_div = d3.select("#network").insert("div", ":first-child")
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
      attn_div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      let [blockIdx, headIdx] = nn.parseNodeId(nodeId);
      finalHeatMap.updateBackground(currentFrame.blocks[blockIdx].attention[headIdx]);
    })
    .on("mouseleave", function () {
      selectedNodeId = null;
      attn_div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      // heatMap.updateBackground(boundary[nn.getOutputNode(network).id]);
    });
  let attnHeatMap = new HeatMap(RECT_SIZE, currentConfig.vocabulary.length, xDomain,
    xDomain, attn_div, { noSvg: true });
  attn_div.datum({ heatmap: attnHeatMap, id: nodeId });


  let output_div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}-out`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + RECT_SIZE + 3}px`,
      top: `${y + 3}px`
    })
  let outputHeatMap = new HeatMap(RECT_SIZE, currentConfig.vocabulary.length, xDomain,
    xDomain, output_div, { noSvg: true });
  output_div.datum({ heatmap: outputHeatMap, id: `out_${nodeId}` });


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
  let maxY = nodeIndexScale(currentConfig.vocabulary.length, TOKEN_SIZE);
  currentConfig.vocabulary.forEach((nodeId, i) => {
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
  let value = (nodeOrLink as nn.Link).weight
  let name = "Weight";
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
  let dimension = RECT_SIZE * 2;
  if (currentConfig.vocabulary.includes(input.source.id)) {
    dimension = TOKEN_SIZE;
  }
  let datum = {
    source: {
      y: source.cx + dimension / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE,
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

function updateUI() {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  let selectedId = selectedNodeId != null ?
    selectedNodeId : nn.getOutputNode(network).id;

  d3.select("#network").selectAll("div.canvas")
    .each(function (data: { heatmap: HeatMap, id: string }) {
      if (data.id.startsWith("out_")) {
        let [blockIdx, headIdx] = nn.parseNodeId(data.id.slice(4));
        data.heatmap.updateBackground(currentFrame.blocks[blockIdx].output[headIdx])
      }
      else {
        let [blockIdx, headIdx] = nn.parseNodeId(data.id);
        data.heatmap.updateBackground(currentFrame.blocks[blockIdx].attention[headIdx])
      }
    });


  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function logHumanReadable(n: number): string {
    return Math.log(n).toFixed(2);
  }

  d3.select("#loss-train").text(logHumanReadable(currentFrame.lossTrain));
  d3.select("#loss-test").text(logHumanReadable(currentFrame.lossTest));
  d3.select("#epoch-number").text(addCommas(zeroPad(currentFrame.epoch)));
}

function step(count: number): void {
  state.currentFrameIdx += count;
  if (state.currentFrameIdx >= frames.length) {
    state.currentFrameIdx = frames.length - 1
    player.pause();
  }
  else if (state.currentFrameIdx < 0) {
    state.currentFrameIdx = 0
  }
  currentFrame = frames[state.currentFrameIdx]
  nn.updateWeights(network, currentFrame.blocks[0].attention, null)
  lineChart.setCursor(state.currentFrameIdx)
  updateUI();
}
player.onTick(step)

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
  state.serialize();
  player.pause();
  state.currentFrameIdx = 0
  currentFrame = frames[state.currentFrameIdx]
  lineChart.setData(frames.map(frame => [frame.lossTest, frame.lossTrain]));
  lineChart.setCursor(state.currentFrameIdx)

  // Make a simple network.
  d3.select("#heatmap").selectAll("div").remove();
  finalHeatMap = new HeatMap(300, currentConfig.vocabulary.length, xDomain, xDomain, d3.select("#heatmap"),
    { showAxes: true });
  let shape = new Array(currentConfig.n_blocks).fill(currentConfig.n_heads)
  network = nn.buildNetwork(shape, currentConfig.vocabulary);
  drawNetwork(network);
  updateUI();
};

function initialLoad() {
  fetch('./data/contents.json')
    .then(response => response.json())
    .then(data => {
      experiments = data;
      console.log("Experiment list", experiments)
      let exp_selector = d3.select("#experiment")
      for (let exp_name of experiments) {
        exp_selector.append("option").text(exp_name).attr("value", exp_name)
        fetch(`./data/${exp_name}.json`)
          .then(response => response.json())
          .then(data => {
            expTags[exp_name] = data
          })
      }
      console.log("All Experiment Tags", expTags)
      d3.select("#seed").property("max", expTags[state.experiment].tags.length)
    })
    .catch(error => console.log(error));
}

function loadData(filetag) {
  console.log("Loading tag:", filetag)
  fetch(`./data/${state.experiment}__${filetag}__config.json`)
    .then(response => response.json())
    .then(data => {
      console.log(`Experiment params for '${filetag}'`, data)
      currentConfig = data;
    })
    .catch(error => console.log(error));

  fetch(`./data/${state.experiment}__${filetag}__frames.json`)
    .then(response => response.json())
    .then(data => {
      console.log("Frames", data)
      frames = data;
      nn.updateWeights(network, currentFrame.blocks[0].attention, null)
      reset();
    })
    .catch(error => console.log(error));
}

makeGUI();
initialLoad();
loadData(state.currentTag);
reset()