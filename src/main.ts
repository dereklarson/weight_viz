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
import { colorScale, HeatMap } from "./heatmap";
import { LineChart } from "./linechart";
import * as nn from "./nn";
import { Player } from "./player";
import { State } from "./state";
import './style.css';

/** Define a global Objects for clear organization of variables
 *  gd -- Global data: the loaded experiment data 
 *  gc -- Global components: heatmaps, Player controls
 *  gs -- Global settings, 
*/
interface GlobalData {
  experiments: { [key: string]: { name: string, tags: string[], notes: string } },
  tagGlossary: { [key: string]: string },
  frames: nn.WFrame[],
  currentConfig: nn.TransformerConfig,
  currentFrame: nn.WFrame,
}

let gd: GlobalData = {
  experiments: {},
  tagGlossary: undefined,
  frames: undefined,
  currentConfig: undefined,
  currentFrame: undefined,
}

interface GlobalComponents {
  player: Player,
  lineChart: LineChart,
  loaderTimer: NodeJS.Timeout,
  residuals: HeatMap[],
  inspectHeatMap: HeatMap,
  resultHeatMap: HeatMap,
}

let gc: GlobalComponents = {
  player: new Player(),
  lineChart: undefined,
  loaderTimer: undefined,
  residuals: [],
  resultHeatMap: undefined,
  inspectHeatMap: undefined,
}

interface GlobalSettings {
  activeTokenId: string,
  activeNodeId: string,
  activeVocab: string[],
  inputIdxs: number[],
  residualIds: string[],
  paramTabs: string[]
  transformer: nn.Node[][],
  maxVocab: number,
  maxTokens: number,
  maxHeads: number,
  tokenSize: number,
  nodeSize: number,
  tokenColWidth: number,
  networkSVGHeight: number,
}

let gs: GlobalSettings = {
  activeTokenId: null,
  activeNodeId: "0_0_0",
  activeVocab: undefined,
  inputIdxs: undefined,
  residualIds: undefined,
  paramTabs: ["model", "data", "train"],
  transformer: null,
  maxVocab: 10,
  maxTokens: 30,
  maxHeads: 5,
  tokenSize: 20,
  nodeSize: 60,
  tokenColWidth: 118,
  networkSVGHeight: 450,
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
  data: ["operation", "seed", "training_fraction", "value_range", "dist_style", "value_count", "use_operators"],
  model: ["d_embed", "d_head", "d_mlp", "n_ctx", "n_heads", "n_blocks", "use_position"],
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

function updateExpHover(display: boolean, coordinates?: [number, number]) {
  let expNotes = d3.select("#exp-notes");
  if (!display) {
    expNotes.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  expNotes.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
}

function addConfigurationOptions() {
  let configuration = d3.select("#configuration")
  configuration.selectAll("option").remove()
  gd.experiments[state.experiment].tags.forEach(tag => {
    let name = tag === "default" ? "Default" : parseTag(tag)
    configuration.append("option").text(name).attr("value", tag)
  })
}

function makeGUI() {
  /* Two dropdown menus to select the Experiment and Configuration */
  let configuration = d3.select("#configuration").on("input", function () {
    state.currentTag = this.value;
    loadData(state.experiment, state.currentTag);
  });

  d3.select("#experiment")
    .on("change", function () {
      state.experiment = this.value;
      state.currentTag = gd.experiments[state.experiment].tags[0];
      addConfigurationOptions()
      d3.select("#exp-notes").select(".text")
        .text(gd.experiments[state.experiment].notes);
      loadData(state.experiment, state.currentTag);
    });

  d3.select("#info-icon")
    .on("mouseenter", function () {
      updateExpHover(true, d3.mouse(this));
    }).on("mouseleave", function () {
      updateExpHover(null)
    })

  /* The Parameters table: Tab row at top */
  function setTab(tabName: string) {
    d3.select(`#${state.currentTab}Tab`).classed("active", false);
    state.currentTab = tabName;
    d3.select(`#${tabName}Tab`).classed("active", true);
    setExperimentalParams()
    state.serialize();
  }

  gs.paramTabs.forEach(name =>
    d3.select(`#${name}Tab`).on("click", () => setTab(name))
  )

  /* The Player controls: play/pause, reset, and step left or right */
  d3.select("#play-pause-button").on("click", () => {
    // If play is pressed at the last frame, reset to zero automatically
    if (state.currentFrameIdx === gd.frames.length - 1) setFrame(0)
    gc.player.playOrPause()
  });

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

  function setContextState() {
    d3.select("#residual-box").classed("hidden", !state.useContext)
    d3.select("#vocab-bank").classed("hidden", !state.useContext)
    d3.selectAll(".vocab-label").classed("hidden", state.useContext)
    d3.selectAll(".context-label").classed("hidden", !state.useContext)
  }

  /* Context checkbox switches between token and context modes, shows residual */
  let useContext = d3.select("#use-context").on("change", function () {
    state.useContext = this.checked;
    state.serialize();
    setContextState()
    redraw()
    updateUI();
  });
  useContext.property("checked", state.useContext);
  setContextState()

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
    .domain([0, 0.8])
    .range([1, 10])
    .clamp(true);

  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inLinks.length; j++) {
        let link = node.inLinks[j];
        let marker = "markerArrow"
        let dashOffset = -state.currentFrameIdx / 3
        let stroke = colorScale(link.weight);
        if (gs.activeTokenId === link.source.id && gs.activeNodeId.startsWith(link.dest.id)) {
          stroke = "#777777";
        }
        if (gs.activeTokenId !== null && gs.activeTokenId !== link.source.id) {
          marker = "revMarkerArrow";
          dashOffset = -dashOffset;
        }
        container.select(`#link${link.source.id}-${link.dest.id}`)
          .attr({
            "marker-start": `url(#${marker})`
          })
          .style({
            "stroke-dashoffset": dashOffset,
            "stroke-width": linkWidthScale(Math.abs(link.weight)),
            "stroke": stroke,
          })
          .datum(link);
      }
    }
  }
}
function drawVocabSet(container, position: number[]) {
  let width = 25;
  let height = 25;

  var rows = Math.max(2, Math.floor(gs.activeVocab.length / 5))
  var cols = Math.ceil(gs.activeVocab.length / rows)
  let coords = gs.activeVocab.map((_, idx) => [(idx % cols) * width, Math.floor(idx / cols) * height])

  function makeClickCallback(tokenIdx: number) {
    return function () {
      state.context.push(tokenIdx)
      if (state.context.length > gd.currentConfig.n_ctx) state.context.shift()
      d3.select("#context").text(`${state.context}`)
      redraw()
      updateUI()
    }
  }

  gs.activeVocab.forEach((tokenName, idx) => {
    var nodeGroup = container.append("g")
      .attr({
        "id": `vocab-${idx}`,
        "transform": `translate(${coords[idx][0]},${coords[idx][1]})`
      })
      .on("click", makeClickCallback(idx))

    nodeGroup.append("rect")
      .attr({ width, height, "rx": 6, "ry": 6, "fill": "lightgrey" })

    var text = nodeGroup.append("text").attr({
      class: "vocab-label",
      x: width / 2,
      y: height / 2,
      "dominant-baseline": "middle",
    });
    text.append("tspan").text(tokenName)
  })

  return rows * height
}

function drawTokenNode(container, node: nn.Node) {
  let [x, y] = node.position
  let cols = gd.currentConfig.d_embed

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${node.id}`,
      "transform": `translate(${x},${y})`
    })

  let text = nodeGroup.append("text").attr({
    class: "main-label",
    x: -10,
    y: gs.tokenSize / 2,
    "text-anchor": "end",
    "dominant-baseline": "middle"
  });
  text.append("tspan").text(node.label)

  function makeClickCallback(canvas, _nodeId: string) {
    if (state.useContext && !_nodeId.includes("Vocab")) return () => null
    else if (state.useContext && _nodeId.includes("Vocab")) return function () {
      state.context.push(parseInt(node.id))
      if (state.context.length > gd.currentConfig.n_ctx) state.context.shift()
      d3.select("#context").text(`${state.context}`)
      redraw()
    }
    else return function () {
      d3.select(`#canvas-${state.selectedTokenId}`).classed("active", false);
      state.selectedTokenId = state.selectedTokenId == _nodeId ? null : _nodeId
      canvas.classed("active", _nodeId == state.selectedTokenId);
      updateUI()
    }
  }

  function makeHoverCallback(canvas, _nodeId: string, entering: boolean) {
    return function () {
      gs.activeTokenId = entering ? _nodeId : state.selectedTokenId;
      canvas.classed("hovered", entering);
      updateUI()
    }
  }

  let canvas = d3.select("#network").insert("div", ":first-child")
  canvas
    .attr({
      "id": `canvas-${node.id}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 2}px`,
      top: `${y + 2}px`
    })
    .on("click", makeClickCallback(canvas, node.id))
    .on("mouseenter", makeHoverCallback(canvas, node.id, true))
    .on("mouseleave", makeHoverCallback(canvas, node.id, false))
  let tokenHeatMap = new HeatMap(canvas, 1, cols, node.shape, { noSvg: true });
  canvas.datum({ heatmap: tokenHeatMap, id: node.id });
}

function drawNode(container, node: nn.Node) {
  let [x, y] = node.position
  let rows = state.useContext ? gd.currentConfig.d_embed : gd.currentConfig.n_vocab;
  let cols = rows;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${node.id}`,
      "transform": `translate(${x},${y})`
    });

  function makeClickCallback(canvas, _nodeId: string) {
    return function () {
      state.selectedNodeId = _nodeId === state.selectedNodeId ? null : _nodeId
      d3.select("#network").selectAll("div.canvas").classed("active", false)
      if (state.selectedNodeId) canvas.classed("active", true);
      updateUI()
    }
  }

  function makeHoverCallback(canvas, _nodeId: string, entering: boolean) {
    return function () {
      gs.activeNodeId = entering ? _nodeId : state.selectedNodeId || _nodeId;
      canvas.classed("hovered", entering);
      updateUI()
    }
  }

  // Draw the node's attention and output canvases.
  for (var isOutput of [false, true]) {
    let id = `${node.id}_${+isOutput}`
    let canvas = d3.select("#network").insert("div", ":first-child")
    let padRight = isOutput ? gs.nodeSize + 5 : 0
    canvas
      .attr({
        "id": `canvas-${id}`,
        "class": "canvas"
      })
      .style({
        position: "absolute",
        left: `${x + 5 + padRight}px`,
        top: `${y + 3}px`
      })
      .on("click", makeClickCallback(canvas, id))
      .on("mouseenter", makeHoverCallback(canvas, id, true))
      .on("mouseleave", makeHoverCallback(canvas, id, false))
    let heatmap = new HeatMap(canvas, rows, cols, [gs.nodeSize, gs.nodeSize], { noSvg: true });
    canvas.datum({ heatmap, id });
  }
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function updateWeightsHover(display: boolean, link?: nn.Link,
  coordinates?: [number, number]) {
  let weightsHover = d3.select("#attn-weights");
  if (!display) {
    weightsHover.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  let value = (link as nn.Link).weight
  weightsHover.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  weightsHover.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
}

function drawLink(container, link: nn.Link, offset: number) {
  let line = container.insert("path", ":first-child");

  // We add a small offset to the destination y coordinate, so the links are distinct
  // as they intersect with the boundary of the destination node.
  let datum = {
    source: {
      y: link.source.position[0] + link.source.shape[0],
      x: link.source.position[1] + link.source.shape[1] / 2
    },
    target: {
      y: link.dest.position[0],
      x: link.dest.position[1] + link.dest.shape[1] / 2 + offset
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + link.source.id + "-" + link.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function () {
      updateWeightsHover(true, link, d3.mouse(this));
    }).on("mouseleave", function () {
      updateWeightsHover(null);
    });
  return line;
}

function drawNetwork(network: nn.Node[][]): void {
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.inspection").node() as HTMLDivElement;
  let cf = d3.select(".column.tokens").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: { [id: string]: { cx: number, cy: number } } = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let layerScale = d3.scale.ordinal<number, number>()
    .domain(d3.range(1, numLayers))
    .rangePoints([gs.tokenColWidth, width - gs.nodeSize], 0.7);
  let xStep = Math.floor(width / network.length)
  let nodeXScale = (nodeIndex: number) => nodeIndex * xStep;
  let nodeYScale = (nodeIndex: number, yStep: number) => nodeIndex * yStep;

  // Draw the input layer.
  var tokenIndent = 20;
  let yStart = state.useContext ? drawVocabSet(container, [0, 0]) + 80 : 30

  gs.inputIdxs.forEach((tokenIdx, i) => {
    let cy = yStart + nodeYScale(i, gs.tokenSize + 20);
    let node = network[0][i];
    node.position = [tokenIndent, cy]
    node.shape = [gs.tokenSize * gd.currentConfig.d_embed, gs.tokenSize]
    if (node.shape[0] > 100) {
      node.shape[1] /= node.shape[0] / 100
      node.shape[0] = 100
    }
    drawTokenNode(container, node);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers; layerIdx++) {
    let cx = nodeXScale(layerIdx);
    for (let i = 0; i < network[layerIdx].length; i++) {
      let node = network[layerIdx][i];
      let cy = nodeYScale(i, gs.nodeSize + 25);
      node.position = [cx, cy];
      node.shape = [2 * gs.nodeSize + 8, gs.nodeSize]
      drawNode(container, node);

      for (let j = 0; j < node.inLinks.length; j++) {
        var offset = -node.inLinks.length / 2 + j
        drawLink(container, node.inLinks[j], offset)
      }
    }
  }

  // Adjust the height of the svg to meet the network needs
  svg.attr("height", gs.networkSVGHeight);

  // Adjust the height of the features column.
  let height = getRelativeHeight(d3.select("#network"))
  d3.select(".column.features").style("height", height + "px");
}

function updateUI() {
  state.serialize();

  // Update the links
  nn.updateWeights(gs.transformer, gd.currentFrame.blocks[0].attention, gs.activeTokenId, gs.activeNodeId)
  updateWeightsUI(gs.transformer, d3.select("g.core"));

  /** Update all heatmaps **/
  // Heatmaps for each node: embeddings and attention/output patterns.
  let frameData;
  d3.select("#network").selectAll("div.canvas")
    .each(function (data: { heatmap: HeatMap, id: string }) {
      var [blockIdx, headIdx, isOut] = nn.parseNodeId(data.id);
      // No headIdx indicates it is an input node
      if (headIdx === undefined) {
        frameData = [gd.currentFrame.embedding[gs.inputIdxs[blockIdx]]]
      }
      else {
        blockKey = isOut ? "output" : "attention"
        if (state.useContext) blockKey = isOut ? "ov" : "qk"
        frameData = gd.currentFrame.blocks[blockIdx][blockKey][headIdx]
        // Contextual QK matrix should be masked for clarity
        if (state.useContext && !isOut) frameData = nn.maskAndScale(frameData, 1)
      }
      data.heatmap.updateGraph(frameData)
    });

  // Update the inspection heatmap
  let [blockIdx, headIdx, isOut] = nn.parseNodeId(gs.activeNodeId);
  let blockKey = (isOut === 1) ? "output" : "attention"
  if (state.useContext) blockKey = isOut ? "ov" : "qk"
  frameData = gd.currentFrame.blocks[blockIdx][blockKey][headIdx]
  if (state.useContext && !isOut) frameData = nn.maskAndScale(frameData, 1)
  gc.inspectHeatMap.updateGraph(frameData);

  // Update the residual heatmaps
  if (state.useContext && state.context.length == gd.currentConfig.n_ctx) {
    var forward = nn.forward(state.context, gd.currentFrame, gd.currentConfig)
    console.log("Forward pass", forward)
    gs.residualIds.map((id, idx) => gc.residuals[idx].updateGraph(forward[id]))
    gc.resultHeatMap.updateGraph(forward.unembed);
    d3.select("#output-value").text(forward.result);
  }

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
  d3.select("#exp-notes").select(".text").text(gd.experiments[state.experiment].notes);
  d3.select("#configuration").property("value", state.currentTag)
  d3.select(`#${state.currentTab}Tab`).classed("active", true);
  d3.select("#scrubber").property('value', state.currentFrameIdx)
  d3.select("#scrubber").node().dispatchEvent(new CustomEvent('change'))
  d3.select("#loss-train").text(logHumanReadable(gd.currentFrame.lossTrain));
  d3.select("#loss-test").text(logHumanReadable(gd.currentFrame.lossTest));
  d3.select("#inspect-label").text(`Head: ${blockIdx}.${headIdx} ${blockKey}`);
  d3.select("#accuracy-test").text((gd.currentFrame.accuracyTest * 100).toFixed(0));
  d3.select("#epoch-number").text(addCommas(zeroPad(gd.currentFrame.epoch)));
}

function setFrame(frameIdx: number): void {
  if (frameIdx < 0) frameIdx = Math.max(0, gd.frames.length + frameIdx)
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

function redraw() {
  var ccfg = gd.currentConfig
  var sliceEnd = state.useContext ? gs.maxTokens : gs.maxVocab
  gs.activeVocab = ccfg.vocabulary.slice(0, sliceEnd)
  gs.inputIdxs = state.useContext ? state.context : gs.activeVocab.map((_, idx) => idx)

  let rows = state.useContext ? ccfg.d_embed : ccfg.n_vocab;
  let cols = rows;
  gc.lineChart = new LineChart(d3.select("#linechart"), ["#777", "black"]);
  gc.lineChart.setData(gd.frames.map(frame => [frame.lossTest, frame.lossTrain]));
  gc.inspectHeatMap = new HeatMap(d3.select("#heatmap"), rows, cols, [300, 300], { showAxes: true });
  //Add residuals for pre and post head blocks
  gs.residualIds = ["position", "embedding"]
  d3.select("#residual-blocks").selectAll("div").remove()
  for (let idx = 0; idx <= ccfg.n_blocks; idx++) {
    gs.residualIds.push(`resblock${idx}`)
    var div = d3.select("#residual-blocks").insert("div")
    div.insert("p").text(`Block ${idx}`)
    div.insert("div").attr({ id: `resblock${idx}`, class: "residual-canvas" })
  }
  gc.residuals = gs.residualIds.map((id) =>
    new HeatMap(d3.select(`#${id}`), ccfg.n_ctx, ccfg.d_embed, [100, 0], { maxHeight: 60 })
  )
  gc.resultHeatMap = new HeatMap(d3.select("#unembed"), ccfg.n_ctx, ccfg.n_vocab, [150, 0])
  var networkShape = new Array(ccfg.n_blocks).fill(ccfg.n_heads)
  var inputs = gs.inputIdxs.map(tokenIdx => gs.activeVocab[tokenIdx])
  gs.transformer = nn.buildNetwork(networkShape, inputs, gs.maxHeads);
  drawNetwork(gs.transformer);
};

/** Loads the static data listing all experiments and tag values */
function initialLoad() {
  fetch('./data/tag_glossary.json')
    .then(response => response.json())
    .then(data => gd.tagGlossary = data)

  return fetch('./data/contents.json')
    .then(response => response.json())
    .then(contents => {
      let exp_selector = d3.select("#experiment")
      let res = contents.map((exp_name) =>
        fetch(`./data/${exp_name}.json`)
          .then(response => response.json())
          .then(data => {
            gd.experiments[exp_name] = data
            exp_selector.append("option").text(data.name).attr("value", exp_name)
          })
          .catch(error => console.log(`Couldn't load ${exp_name}.json`))
      )
      Promise.all(res).then(() =>
        console.log("All Experiments", gd.experiments)
      )
    })
    .catch(error => console.log(error));
}

/** Load an experiment's full data */
function loadData(experiment: string, filetag: string) {
  gc.loaderTimer = setTimeout(() => d3.select("#loader").classed("hidden", false), 1000)
  console.log("Loading experiment/tag:", experiment, filetag)
  fetch(`./data/${state.experiment}__${filetag}__config.json`)
    .then(response => response.json())
    .then(data => {
      console.log(`${experiment}_${filetag} Config`, data)
      gd.currentConfig = data;
      setExperimentalParams()
    })
    .catch(error => console.log(error));

  fetch(`./data/${state.experiment}__${filetag}__frames.json`)
    .then(response => response.json())
    .then(data => {
      console.log("Frames", data)
      gd.frames = data
      gs.activeTokenId = null
      state.selectedTokenId = null
      state.selectedNodeId = "0_0_0"
      state.context = []
      clearTimeout(gc.loaderTimer)
      d3.select("#loader").classed("hidden", true)
      d3.select("#main-part").classed("hidden", false)
      redraw();
      addConfigurationOptions()
      setFrame(-1);
      updateUI();
    })
    .catch(error => console.log(error));
}

makeGUI();
initialLoad().then(() => loadData(state.experiment, state.currentTag))