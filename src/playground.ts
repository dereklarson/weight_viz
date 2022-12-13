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

const TOKEN_SIZE = 20;
const RECT_SIZE = 60;

let mainWidth;
let player = new Player();

// State contains variables we can load from the URL
let state = State.deserializeState();

// Supplementary state variables that won't be serialized
let selectedTokenId: string = null;
let selectedNodeId: string = "0_0";

/*** Variables that contain loaded data ***/
// A supplied glossary for parameter abbrevations
let tagGlossary: { [key: string]: string } = {}
function parseTag(tag: string) {
  let terms = []
  for (let pair of tag.split("_")) {
    let [key, val] = pair.split("@")
    terms.push(`${tagGlossary[key]}=${val}`)
  }
  return terms.join(", ")
}

let transformer: nn.Node[][] = null;
let experiments = { sample: { name: "Sample", tags: ["default"] } }
let currentConfig: nn.TransformerConfig
let currentFrame: nn.Frame
let frames: nn.Frame[]

let xDomain: [number, number] = [-6, 6];
let lineChart;
let residualHeatMap;
let positionalHeatMap;
let finalHeatMap =
  new HeatMap(300, 10, 10, xDomain, xDomain, d3.select("#heatmap"),
    { showAxes: true });
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#f59322", "#e8eaeb", "#0877bd"])
  .clamp(true);

function titleCase(str: string) {
  return str.split("_").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")
}

let paramKeys = {
  data: ["seed", "training_fraction", "value_range", "dist_style", "value_count"],
  model: ["d_embed", "d_head", "d_mlp", "n_ctx", "n_heads", "n_blocks"],
  train: ["learning_rate", "weight_decay", "n_epochs"],
}

function setExperimentalParams() {
  let expParams = d3.select("#experimental-params").select("tbody")
  expParams.selectAll("tr").remove()
  for (var key in currentConfig) {
    if (!paramKeys[state.currentTab].includes(key)) continue;
    var row = expParams.append("tr").classed("row", true)
    row.append("td").classed("datum", true).text(titleCase(key))
    row.append("td").classed("datum", true).text(currentConfig[key])
  }
}

function makeGUI() {
  /* Two dropdown menus to select the Experiment and Configuration */
  let configuration = d3.select("#configuration").on("input", function () {
    state.currentTag = this.value;
    loadData(state.experiment, state.currentTag);
  });

  d3.select("#experiment").on("change", function () {
    state.experiment = this.value;
    state.currentTag = experiments[state.experiment].tags[0];
    configuration.selectAll("option").remove()
    experiments[state.experiment].tags.forEach(tag => {
      let name = tag === "default" ? "Default" : parseTag(tag)
      configuration.append("option").text(name).attr("value", tag)
    })
    loadData(state.experiment, state.currentTag);
  });

  function setTab(tabName: string) {
    d3.select(`#${state.currentTab}Tab`).classed("active", false);
    state.currentTab = tabName;
    d3.select(`#${tabName}Tab`).classed("active", true);
    setExperimentalParams()
  }

  d3.select("#modelTab").on("click", () => setTab("model"));
  d3.select("#dataTab").on("click", () => setTab("data"));
  d3.select("#trainTab").on("click", () => setTab("train"));

  /* The Player controls: play/pause, reset, and step left or right */
  d3.select("#play-pause-button").on("click", () => player.playOrPause());

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#reset-button").on("click", () => {
    player.pause();
    setFrame(0)
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    step(1);
  });

  d3.select("#prev-step-button").on("click", () => {
    player.pause();
    step(-1);
  });

  /* Scrubbing slider allows easy frame navigation */
  d3.select("#scrubber").on("input", function () { setFrame(parseInt(this.value)) })

  /* Context checkbox switches between token and context modes, shows residual */
  let useContext = d3.select("#use-context").on("change", function () {
    state.useContext = this.checked;
    state.serialize();
    d3.select("#residual-box").classed("hidden", !this.checked)
    redraw()
  });
  useContext.property("checked", state.useContext);
  d3.select("#residual-box").classed("hidden", !state.useContext)

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

function drawTokenNode(cx: number, cy: number, nodeId: string,
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
      selectedTokenId = nodeId;
      rect.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateUI();
    })
    .on("mouseleave", function () {
      selectedTokenId = null;
      rect.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateUI();
    })
    .on("click", function () {
      state.tokenState[nodeId] = !state.tokenState[nodeId];
      state.context.push(parseInt(nodeId))
      if (state.context.length > currentConfig.n_ctx) {
        state.context.shift()
      }
      d3.select("#context").text(`${state.context}`)
      redraw()
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
      attn_div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      // heatMap.updateBackground(boundary[nn.getOutputNode(network).id]);
    });
  let attnHeatMap = new HeatMap(RECT_SIZE,
    currentConfig.n_vocab, currentConfig.n_vocab,
    xDomain, xDomain, attn_div, { noSvg: true });
  attn_div.datum({ heatmap: attnHeatMap, id: nodeId });


  let output_div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}-out`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + RECT_SIZE + 8}px`,
      top: `${y + 3}px`
    })
  let outputHeatMap = new HeatMap(RECT_SIZE, currentConfig.n_vocab, currentConfig.n_vocab,
    xDomain, xDomain, output_div, { noSvg: true });
  output_div.datum({ heatmap: outputHeatMap, id: `out_${nodeId}` });


}

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
    drawTokenNode(cx, cy, nodeId, container);
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

function updateHoverCard(display: boolean, link?: nn.Link,
  coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (!display) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  let value = (link as nn.Link).weight
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
}

function drawLink(
  input: nn.Link, node2coord: { [id: string]: { cx: number, cy: number } },
  network: nn.Node[][], container,
  isFirst: boolean, index: number, length: number) {
  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let dimension = RECT_SIZE * 2 + 10;
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
      updateHoverCard(true, input, d3.mouse(this));
    }).on("mouseleave", function () {
      updateHoverCard(null);
    });
  return line;
}

function updateUI() {
  state.serialize();

  // Update the links
  nn.updateWeights(transformer, currentFrame.blocks[0].attention, selectedTokenId)
  updateWeightsUI(transformer, d3.select("g.core"));

  // Update all heatmaps
  let [blockIdx, headIdx] = nn.parseNodeId(selectedNodeId);
  finalHeatMap.updateBackground(currentFrame.blocks[blockIdx].attention[headIdx]);

  positionalHeatMap.updateBackground(currentFrame.pos_embed);
  if (state.useContext) {
    let residual = new Array(currentConfig.n_ctx).fill(0).map(_ => new Array(currentConfig.d_embed).fill(0))
    state.context.forEach((tokenIdx, ctxIdx) => residual[ctxIdx] = currentFrame.embedding[tokenIdx])
    residualHeatMap.updateBackground(residual);
  }

  // Update all attention and output patterns for nodes.
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


  d3.select("#experiment").property("value", state.experiment)
  d3.select(`#${state.currentTab}Tab`).classed("active", true);
  d3.select("#configuration").property("value", state.currentTag)
  d3.select("#scrubber").property('value', state.currentFrameIdx)
  d3.select("#scrubber").node().dispatchEvent(new CustomEvent('change'))
  d3.select("#loss-train").text(logHumanReadable(currentFrame.lossTrain));
  d3.select("#loss-test").text(logHumanReadable(currentFrame.lossTest));
  d3.select("#epoch-number").text(addCommas(zeroPad(currentFrame.epoch)));
}

function setFrame(frameIdx: number): void {
  state.currentFrameIdx = frameIdx;
  currentFrame = frames[state.currentFrameIdx]
  lineChart.setCursor(state.currentFrameIdx)
  updateUI();
}

function step(count: number): void {
  let frameIdx = state.currentFrameIdx + count;
  frameIdx = Math.max(0, Math.min(frames.length - 1, frameIdx))
  if (frameIdx === frames.length - 1) {
    player.pause();
  }
  setFrame(frameIdx)
  d3.select("#scrubber").property("value", state.currentFrameIdx);
  d3.select("#scrubber").node().dispatchEvent(new CustomEvent('change'))
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

function redraw() {
  lineChart = new LineChart(d3.select("#linechart"), ["#777", "black"]);
  lineChart.setData(frames.map(frame => [frame.lossTest, frame.lossTrain]));
  d3.select("#heatmap").selectAll("div").remove();
  finalHeatMap = new HeatMap(300, currentConfig.n_vocab, currentConfig.n_vocab,
    xDomain, xDomain, d3.select("#heatmap"), { showAxes: true });
  d3.select("#residual-input").selectAll("div").remove();
  residualHeatMap =
    new HeatMap(100, currentConfig.n_ctx, currentConfig.d_embed, xDomain, xDomain, d3.select("#residual-input"));
  d3.select("#positional-embedding").selectAll("div").remove();
  positionalHeatMap =
    new HeatMap(100, currentConfig.n_ctx, currentConfig.d_embed, xDomain, xDomain, d3.select("#positional-embedding"));
  let shape = new Array(currentConfig.n_blocks).fill(currentConfig.n_heads)
  transformer = nn.buildNetwork(shape, currentConfig.vocabulary);
  drawNetwork(transformer);
  setFrame(0);
  updateUI();
};

function initialLoad() {
  fetch('./data/tag_glossary.json')
    .then(response => response.json())
    .then(data => tagGlossary = data)

  fetch('./data/contents.json')
    .then(response => response.json())
    .then(contents => {
      let exp_selector = d3.select("#experiment")
      for (let exp_name of contents) {
        fetch(`./data/${exp_name}.json`)
          .then(response => response.json())
          .then(data => {
            experiments[exp_name] = data
            exp_selector.append("option").text(data.name).attr("value", exp_name)
          })
          .catch(error => console.log(error))
      }
      console.log("All Experiments", experiments)
    })
    .catch(error => console.log(error));
}

function loadData(experiment: string, filetag: string) {
  console.log("Loading experiment/tag:", experiment, filetag)
  fetch(`./data/${state.experiment}__${filetag}__config.json`)
    .then(response => response.json())
    .then(data => {
      console.log(`Experiment params for '${filetag}'`, data)
      currentConfig = data;
      setExperimentalParams()
    })
    .catch(error => console.log(error));

  fetch(`./data/${state.experiment}__${filetag}__frames.json`)
    .then(response => response.json())
    .then(data => {
      console.log("Frames", data)
      frames = data;
      selectedTokenId = null
      d3.select("#loader").classed("hidden", true)
      d3.select("#main-part").classed("hidden", false)
      redraw();
    })
    .catch(error => console.log(error));
}

makeGUI();
initialLoad();
loadData(state.experiment, state.currentTag);