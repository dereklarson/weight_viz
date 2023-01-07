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
import { matrix } from './minMathjs';

// Mathjs Matrix type was difficult to extract via minMathjs
type Matrix = any;

export interface HeatMapSettings {
  [key: string]: any;
  showAxes?: boolean;
  noSvg?: boolean;
}

// Original Orange -> Grey -> Blue scheme
let colorArray = ["#f59322", "#e8eaeb", "#0877bd"]
let tooltipTimer;

export let colorScale = d3.scale.linear<string, string>()
  .domain([-1, 0, 1])
  .range(colorArray)
  .clamp(true);

// Adjust the color legend
d3.select("#gradient").selectAll("stop").each(function (d, idx) {
  d3.select(this).attr("stop-color", colorArray[idx])
})

function makeMatrixTable(container, matrix) {
  container.selectAll("table").remove()
  let table = container.append("table").attr({ "id": "matrix-tooltip" })
  for (var i = 0; i < matrix.size()[0]; i++) {
    let row = table.append("tr")
    for (var j = 0; j < matrix.size()[1]; j++) {
      row.append("td").text(matrix.get([i, j]).toFixed(3))
    }
  }
}

function addTooltip(container, self) {
  let tooltip = d3.select("#network-tooltip").text("Matrix contents")

  function makeHoverCallback(_entering: boolean) {
    return function () {
      clearTimeout(tooltipTimer)
      if (_entering) {
        makeMatrixTable(tooltip, self.data)
        tooltipTimer = setTimeout(() => tooltip.classed("hidden", false), 1000)
      }
      else {
        tooltip.classed("hidden", true)
      }
    }
  }
  container
    .on("mouseenter", makeHoverCallback(true))
    .on("mouseleave", makeHoverCallback(false))
}

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class HeatMap {
  private settings: HeatMapSettings = {
    showAxes: false,
    noSvg: false,
    maxHeight: null,
  };
  private nRow: number;
  private nCol: number;
  private colorScale;
  private canvas;
  private svg;
  public data;

  constructor(
    width: number, nRow: number, nCol: number,
    container, userSettings?: HeatMapSettings) {
    this.nRow = nRow;
    this.nCol = nCol;
    this.colorScale = colorScale
    let height = (nRow / nCol) * width

    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }
    let padding = this.settings.showAxes ? 20 : 0;
    if (this.settings.maxHeight !== null && height > this.settings.maxHeight) {
      var ratio = this.settings.maxHeight / height;
      height = this.settings.maxHeight;
      width = width * ratio;
    }

    // Clean out any existing content
    container.selectAll("div").remove()

    container = container.append("div")
      .style({
        width: `${width}px`,
        height: `${height}px`,
        position: "relative",
        top: `-${padding}px`,
        left: `-${padding}px`
      });
    this.canvas = container.append("canvas")
      .attr("width", nCol)
      .attr("height", nRow)
      .style("width", (width - 2 * padding) + "px")
      .style("height", (height - 2 * padding) + "px")
      .style("position", "absolute")
      // 'image-rendering: pixelated' avoids blurring between datapoints on scaling
      .style("image-rendering", "pixelated");

    addTooltip(container, this)

    if (!this.settings.noSvg) {
      this.svg = container.append("svg").attr({
        "width": width,
        "height": height
      }).style({
        // Overlay the svg on top of the canvas.
        "position": "absolute",
        "left": "0",
        "top": "0"
      }).append("g")
        .attr("transform", `translate(${padding},${padding})`);

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    if (this.settings.showAxes) {
      // For the inspection heatmap, we need the offset provided by padding
      this.canvas
        .style("top", `${padding}px`)
        .style("left", `${padding}px`)
      var xDim = width - 2 * padding;
      var yDim = height - 2 * padding;
      var xScale = d3.scale.linear()
        .domain([0, nCol - 1])
        .range([0, yDim * (nCol - 1) / nCol]);

      var xAxis = d3.svg.axis()
        .scale(xScale)
        .orient("bottom");

      var yScale = d3.scale.linear()
        .domain([0, nRow - 1])
        .range([0, xDim * (nRow - 1) / nRow]);

      var yAxis = d3.svg.axis()
        .scale(yScale)
        .orient("right");

      this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", `translate(${width / (2 * nCol)},${yDim})`)
        .call(xAxis);

      this.svg.append("g")
        .attr("class", "y axis")
        .attr("transform", `translate(${xDim}, ${height / (2 * nRow)})`)
        .call(yAxis);
    }
  }

  updateGraph(_data: Matrix | number[][]): void {
    let data = matrix(_data)
    this.data = data
    let [dy, dx] = data.size()

    if (dx !== this.nCol || dy !== this.nRow) {
      throw new Error(
        "The provided data matrix must be of size " +
        `nRow(${this.nRow}) X nCol(${this.nCol})`);
    }

    // Compute the pixel colors; scaled by CSS.
    let context = (this.canvas.node() as HTMLCanvasElement).getContext("2d");
    let image = context.createImageData(dx, dy);

    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data.get([y, x]);
        let c = d3.rgb(this.colorScale(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 190;
      }
    }
    context.putImageData(image, 0, 0);
  }
} 