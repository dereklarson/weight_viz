import {Example2D} from "./dataset";

export interface HeatMapSettings {
  [key: string]: any;
  showGridLines?: boolean;
  numGridLines?: number;
  noSvg?: boolean;
}

/** Number of different shades (colors) when drawing a gradient heatmap */
const NUM_SHADES = 30;

/**
 * Draws a heatmap using canvas. Used for showing the learned decision
 * boundary of the classification algorithm. Can also draw data points
 * using an svg overlayed on top of the canvas heatmap.
 */
export class HeatMap {
  private settings: HeatMapSettings = {
    showGridLines: false,
    numGridLines: 5,
    noSvg: false
  };
  private xScale: d3.scale.Linear<number, number>;
  private yScale: d3.scale.Linear<number, number>;
  private numSamples: number;
  private color: d3.scale.Quantize<string>;
  private canvas: d3.Selection<any>;
  private svg: d3.Selection<any>;

  constructor(
      width: number, numSamples: number, xDomain: [number, number],
      yDomain: [number, number], container: d3.Selection<any>,
      userSettings?: HeatMapSettings) {
    this.numSamples = numSamples;
    let height = width;
    if (userSettings != null) {
      // overwrite the defaults with the user-specified settings.
      for (let prop in userSettings) {
        this.settings[prop] = userSettings[prop];
      }
    }

    this.xScale = d3.scale.linear().domain(xDomain).range([0, width]);

    this.yScale = d3.scale.linear().domain(yDomain).range([height, 0]);

    // Get a range of colors.
    let tmpScale = d3.scale.linear<string, string>()
        .domain([0, 0.5, 1])
        .range(["#f59322", "#e8eaeb", "#0877bd"])
        .clamp(true);
    // Due to numerical error, we need to specify
    // d3.range(0, end + small_epsilon, step)
    // in order to guarantee that we will have end/step entries with
    // the last element being equal to end.
    let colors = d3.range(0, 1 + 1E-9, 1 / NUM_SHADES).map(a => {
      return tmpScale(a);
    });
    this.color = d3.scale.quantize<string>()
                     .domain([0, 1])
                     .range(colors);

    this.canvas = container.append("canvas")
                      .attr("width", numSamples)
                      .attr("height", numSamples)
                      .style("width", width + "px")
                      .style("height", height + "px");

    if (!this.settings.noSvg) {
      this.svg = container.append("svg").attr({
          "width": width,
          "height": height
      }).style({
        // Overlay the svg on top of the canvas.
        "position": "absolute",
        "left": "0",
        "top": "0"
      });

      this.svg.append("g").attr("class", "train");
      this.svg.append("g").attr("class", "test");
    }

    if (this.settings.showGridLines) {
      let xAxis = d3.svg.axis()
        .scale(this.xScale)
        .orient("top")
        .innerTickSize(height)
        .outerTickSize(0)
        .ticks(this.settings.numGridLines)
        .tickFormat("");

      this.svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis);

      let yAxis = d3.svg.axis()
        .scale(this.yScale)
        .orient("right")
        .innerTickSize(width)
        .outerTickSize(0)
        .ticks(this.settings.numGridLines)
        .tickFormat("");

      this.svg.append("g").attr("class", "y axis").call(yAxis);
    }
  }

  updateTestPoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select("g.test"), points);
  }

  updatePoints(points: Example2D[]): void {
    if (this.settings.noSvg) {
      throw Error("Can't add points since noSvg=true");
    }
    this.updateCircles(this.svg.select("g.train"), points);
  }

  updateBackground(data: number[][], discretize: boolean): void {
    let dx = data[0].length;
    let dy = data.length;

    if (dx !== this.numSamples || dy !== this.numSamples) {
      throw new Error(
          "The provided data matrix must be of size " +
          "numSamples X numSamples");
    }

    // Compute the pixel colors; scaled by CSS.
    let context = (<HTMLCanvasElement>this.canvas.node()).getContext("2d");
    let image = context.createImageData(dx, dy);

    for (let y = 0, p = -1; y < dy; ++y) {
      for (let x = 0; x < dx; ++x) {
        let value = data[x][y];
        if (discretize) {
          value = (value >= 0.5 ? 1 : 0);
        }
        let c = d3.rgb(this.color(value));
        image.data[++p] = c.r;
        image.data[++p] = c.g;
        image.data[++p] = c.b;
        image.data[++p] = 160;
      }
    }
    context.putImageData(image, 0, 0);
  }

  private updateCircles(container: d3.Selection<any>, points: Example2D[]) {
    // Attach data to initially empty selection.
    let selection = container.selectAll("circle").data(points);

    // Insert elements to match length of points array.
    selection.enter().append("circle").attr("r", 3);

    // Update points to be in the correct position.
    selection
      .attr({
        cx: (d: Example2D) => this.xScale(d.x),
        cy: (d: Example2D) => this.yScale(d.y),
      })
      .style("fill", d => this.color(d.label));

    // Remove points if the length has gone down.
    selection.exit().remove();
  }
}  // Close class HeatMap.

export function reduceMatrix(matrix: number[][], factor: number): number[][] {
  if (matrix.length !== matrix[0].length) {
    throw new Error("The provided matrix must be a square matrix");
  }
  if (matrix.length % factor !== 0) {
    throw new Error("The width/height of the matrix must be divisible by " +
        "the reduction factor");
  }
  let result: number[][] = new Array(matrix.length / factor);
  for (let i = 0; i < matrix.length; i += factor) {
    result[i / factor] = new Array(matrix.length / factor);
    for (let j = 0; j < matrix.length; j += factor) {
      let avg = 0;
      // Sum all the values in the neighborhood.
      for (let k = 0; k < factor; k++) {
        for (let l = 0; l < factor; l++) {
          avg += matrix[i + k][j + l];
        }
      }
      avg /= (factor * factor);
      result[i / factor][j / factor] = avg;
    }
  }
  return result;
}