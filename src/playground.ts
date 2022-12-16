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
import { matrix } from "mathjs";
import { colorScale, HeatMap } from "./heatmap";
import { LineChart } from "./linechart";
import * as nn from "./nn";
import { Player } from "./player";
import { State } from "./state";

const TOKEN_SIZE = 20;
const RECT_SIZE = 60;

/** Define a global Objects for clear organization of variables
 *  gd -- Global data: the loaded experiment data 
 *  gc -- Global components: heatmaps, Player controls
 *  gs -- Global settings, 
*/
interface GlobalData {
  experiments: { [key: string]: { name: string, tags: string[] } },
  tagGlossary: { [key: string]: string },
  frames: nn.WFrame[],
  currentConfig: nn.TransformerConfig,
  currentFrame: nn.WFrame,
}

let gd: GlobalData = {
  experiments: { sample: { name: "Sample", tags: ["default"] } },
  tagGlossary: undefined,
  frames: undefined,
  currentConfig: undefined,
  currentFrame: undefined,
}

interface GlobalComponents {
  player: Player,
  lineChart: LineChart,
  residuals: HeatMap[],
  inspectHeatMap: HeatMap,
  resultHeatMap: HeatMap,
}

let gc: GlobalComponents = {
  player: new Player(),
  lineChart: undefined,
  residuals: [],
  resultHeatMap: undefined,
  inspectHeatMap: undefined,
}

interface GlobalSettings {
  selectedTokenId: string,
  inspectedNodeId: string,
  residualIds: string[],
  transformer: nn.Node[][],
}

let gs: GlobalSettings = {
  selectedTokenId: null,
  inspectedNodeId: "0_0",
  residualIds: ["position", "embedding", "preBlock", "block1"],
  transformer: null,
}

// State contains variables we can load from the URL
let state = State.deserializeState();

// A supplied glossary for parameter abbrevations
function parseTag(tag: string) {
  let terms = []
  for (let pair of tag.split("_")) {
    let [key, val] = pair.split("@")
    terms.push(`${gd.tagGlossary[key]}=${val}`)
  }
  return terms.join(", ")
}

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
  for (var key in gd.currentConfig) {
    if (!paramKeys[state.currentTab].includes(key)) continue;
    var row = expParams.append("tr").classed("row", true)
    row.append("td").classed("datum", true).text(titleCase(key))
    row.append("td").classed("datum", true).text(gd.currentConfig[key])
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
    state.currentTag = gd.experiments[state.experiment].tags[0];
    configuration.selectAll("option").remove()
    gd.experiments[state.experiment].tags.forEach(tag => {
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
  d3.select("#play-pause-button").on("click", () => gc.player.playOrPause());

  gc.player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#reset-button").on("click", () => {
    gc.player.pause();
    setFrame(0)
  });

  d3.select("#next-step-button").on("click", () => {
    gc.player.pause();
    step(1);
  });

  d3.select("#prev-step-button").on("click", () => {
    gc.player.pause();
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

  let usePosEmbed = d3.select("#use-pos-embed").on("change", function () {
    state.usePosEmbed = this.checked;
    state.serialize();
    updateUI()
  });
  usePosEmbed.property("checked", state.usePosEmbed);

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
    drawNetwork(gs.transformer)
    updateUI()
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  const linkWidthScale = d3.scale.linear()
    .domain([0, 5])
    .range([1, 10])
    .clamp(true);

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
  let x = cx - RECT_SIZE / 2;
  let y = cy - TOKEN_SIZE / 2;
  let cols = gd.currentConfig.d_embed

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    })

  let text = nodeGroup.append("text").attr({
    class: "main-label",
    x: -10,
    y: TOKEN_SIZE / 2, "text-anchor": "end"
  });
  text.append("tspan").text(nodeId)

  function makeClickCallback(canvas, _nodeId: string) {
    return function () {
      state.tokenState[_nodeId] = !state.tokenState[_nodeId];
      state.context.push(parseInt(nodeId))
      if (state.context.length > gd.currentConfig.n_ctx) {
        state.context.shift()
      }
      d3.select("#context").text(`${state.context}`)
      if (state.tokenState[_nodeId]) canvas.classed("active", true);
      else canvas.classed("active", false);
    }
  }

  function makeHoverCallback(canvas, _nodeId: string, entering: boolean) {
    return function () {
      gs.selectedTokenId = entering ? _nodeId : state.selectedNodeId || _nodeId;
      canvas.classed("hovered", entering);
      updateUI()
    }
  }

  let canvas = d3.select("#network").insert("div", ":first-child")
  canvas
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("click", makeClickCallback(canvas, nodeId))
    .on("mouseenter", makeHoverCallback(canvas, nodeId, true))
    .on("mouseleave", makeHoverCallback(canvas, nodeId, false))
  let tokenHeatMap = new HeatMap(RECT_SIZE, 1, cols, canvas, { noSvg: true });
  canvas.datum({ heatmap: tokenHeatMap, id: nodeId });
}


function drawNode(cx: number, cy: number, nodeId: string, container, node?: nn.Node) {
  let x = cx - RECT_SIZE;
  let y = cy - RECT_SIZE / 2;
  let rows = gd.currentConfig.n_vocab;
  let cols = rows;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  function makeClickCallback(canvas, _nodeId: string) {
    return function () {
      state.selectedNodeId = _nodeId === state.selectedNodeId ? null : _nodeId
      d3.select("#network").selectAll("div.canvas").classed("active", false)
      if (state.selectedNodeId) canvas.classed("active", true);
    }
  }

  function makeHoverCallback(canvas, _nodeId: string, entering: boolean) {
    return function () {
      gs.inspectedNodeId = entering ? _nodeId : state.selectedNodeId || _nodeId;
      canvas.classed("hovered", entering);
      updateUI()
    }
  }

  // Draw the node's attention and output canvases.
  let attnId = nodeId + "_0";
  let attn_canvas = d3.select("#network").insert("div", ":first-child")
  attn_canvas
    .attr({
      "id": `canvas-${attnId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`
    })
    .on("click", makeClickCallback(attn_canvas, attnId))
    .on("mouseenter", makeHoverCallback(attn_canvas, attnId, true))
    .on("mouseleave", makeHoverCallback(attn_canvas, attnId, false))
  let attnHeatMap = new HeatMap(RECT_SIZE, rows, cols, attn_canvas, { noSvg: true });
  attn_canvas.datum({ heatmap: attnHeatMap, id: attnId });

  var outputId = nodeId + "_1"
  let output_canvas = d3.select("#network").insert("div", ":first-child")
  output_canvas
    .attr({
      "id": `canvas-${outputId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + RECT_SIZE + 8}px`,
      top: `${y + 3}px`
    })
    .on("click", makeClickCallback(output_canvas, outputId))
    .on("mouseenter", makeHoverCallback(output_canvas, outputId, true))
    .on("mouseleave", makeHoverCallback(output_canvas, outputId, false))
  let outputHeatMap = new HeatMap(RECT_SIZE, rows, cols, output_canvas, { noSvg: true });
  output_canvas.datum({ heatmap: outputHeatMap, id: outputId });
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
  let xDim = RECT_SIZE;
  let yDim = TOKEN_SIZE;
  var indent = 30;
  let cx = xDim / 2 + indent;
  let maxY = nodeIndexScale(gd.currentConfig.vocabulary.length, yDim);
  gd.currentConfig.vocabulary.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i, yDim) + yDim / 2;
    drawTokenNode(cx, cy, nodeId, container);
    cy -= 5;
    node2coord[nodeId] = { cx, cy };
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
  if (gd.currentConfig.vocabulary.includes(input.source.id)) {
    dimension = RECT_SIZE;
  }
  let datum = {
    source: {
      y: source.cx + dimension / 2,
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
  nn.updateWeights(gs.transformer, gd.currentFrame.blocks[0].attention, gs.selectedTokenId, state.selectedNodeId)
  updateWeightsUI(gs.transformer, d3.select("g.core"));

  /** Update all heatmaps **/
  let [blockIdx, headIdx, isOut] = nn.parseNodeId(gs.inspectedNodeId);
  var blockKey = isOut ? "output" : "attention"
  gc.inspectHeatMap.updateBackground(matrix(gd.currentFrame.blocks[blockIdx][blockKey][headIdx]));

  if (state.useContext && state.context.length == gd.currentConfig.n_ctx) {
    var forward = nn.forward(state.context, gd.currentFrame, gd.currentConfig)
    // console.log("Forward pass", forward)
    gs.residualIds.map((id, idx) => gc.residuals[idx].updateBackground(forward[id]))
    gc.resultHeatMap.updateBackground(forward.unembed);
  }

  // Update all attention and output patterns for nodes.
  d3.select("#network").selectAll("div.canvas")
    .each(function (data: { heatmap: HeatMap, id: string }) {
      var [blockIdx, headIdx, isOut] = nn.parseNodeId(data.id);
      let frameData;
      if (headIdx === undefined) {
        frameData = [gd.currentFrame.embedding[blockIdx]]
      }
      else {
        var blockKey = isOut ? "output" : "attention"
        frameData = gd.currentFrame.blocks[blockIdx][blockKey][headIdx]
      }
      data.heatmap.updateBackground(matrix(frameData))
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


  d3.select("#context").text(`${state.context}`)
  d3.select("#experiment").property("value", state.experiment)
  d3.select(`#${state.currentTab}Tab`).classed("active", true);
  d3.select("#configuration").property("value", state.currentTag)
  d3.select("#scrubber").property('value', state.currentFrameIdx)
  d3.select("#scrubber").node().dispatchEvent(new CustomEvent('change'))
  d3.select("#loss-train").text(logHumanReadable(gd.currentFrame.lossTrain));
  d3.select("#loss-test").text(logHumanReadable(gd.currentFrame.lossTest));
  d3.select("#epoch-number").text(addCommas(zeroPad(gd.currentFrame.epoch)));
}

function setFrame(frameIdx: number): void {
  state.currentFrameIdx = frameIdx;
  gd.currentFrame = gd.frames[state.currentFrameIdx]
  gc.lineChart.setCursor(state.currentFrameIdx)
  updateUI();
}

function step(count: number): void {
  let frameIdx = state.currentFrameIdx + count;
  frameIdx = Math.max(0, Math.min(gd.frames.length - 1, frameIdx))
  if (frameIdx === gd.frames.length - 1) {
    gc.player.pause();
  }
  setFrame(frameIdx)
  d3.select("#scrubber").property("value", state.currentFrameIdx);
  d3.select("#scrubber").node().dispatchEvent(new CustomEvent('change'))
}
gc.player.onTick(step)

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
  var ccfg = gd.currentConfig
  gc.lineChart = new LineChart(d3.select("#linechart"), ["#777", "black"]);
  gc.lineChart.setData(gd.frames.map(frame => [frame.lossTest, frame.lossTrain]));
  gc.inspectHeatMap = new HeatMap(300, ccfg.n_vocab, ccfg.n_vocab, d3.select("#heatmap"), { showAxes: true });
  gc.residuals = gs.residualIds.map((id) =>
    new HeatMap(100, ccfg.n_ctx, ccfg.d_embed, d3.select(`#${id}`), { maxHeight: 60 })
  )
  gc.resultHeatMap = new HeatMap(100, ccfg.n_ctx, ccfg.n_vocab, d3.select("#unembed"))
  let shape = new Array(gd.currentConfig.n_blocks).fill(gd.currentConfig.n_heads)
  gs.transformer = nn.buildNetwork(shape, gd.currentConfig.vocabulary);
  drawNetwork(gs.transformer);
  setFrame(0);
  updateUI();
};

function initialLoad() {
  fetch('./data/tag_glossary.json')
    .then(response => response.json())
    .then(data => gd.tagGlossary = data)

  fetch('./data/contents.json')
    .then(response => response.json())
    .then(contents => {
      let exp_selector = d3.select("#experiment")
      for (let exp_name of contents) {
        fetch(`./data/${exp_name}.json`)
          .then(response => response.json())
          .then(data => {
            gd.experiments[exp_name] = data
            exp_selector.append("option").text(data.name).attr("value", exp_name)
          })
          .catch(error => console.log(error))
      }
      console.log("All Experiments", gd.experiments)
    })
    .catch(error => console.log(error));
}

function loadData(experiment: string, filetag: string) {
  console.log("Loading experiment/tag:", experiment, filetag)
  fetch(`./data/${state.experiment}__${filetag}__config.json`)
    .then(response => response.json())
    .then(data => {
      console.log(`Experimental params for '${filetag}'`, data)
      gd.currentConfig = data;
      setExperimentalParams()
    })
    .catch(error => console.log(error));

  fetch(`./data/${state.experiment}__${filetag}__frames.json`)
    .then(response => response.json())
    .then(data => {
      console.log("Frames", data)
      gd.frames = data
      gs.selectedTokenId = null
      d3.select("#loader").classed("hidden", true)
      d3.select("#main-part").classed("hidden", false)
      redraw();
    })
    .catch(error => console.log(error));
}

makeGUI();
initialLoad();
loadData(state.experiment, state.currentTag);