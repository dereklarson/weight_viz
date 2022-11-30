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

/** A map between experiment names and specifications. */
export let experiments: { [key: string]: string } = {
  "temp": "temp"
};

export function getKeyFromValue(obj: any, value: any): string {
  for (let key in obj) {
    if (obj[key] === value) {
      return key;
    }
  }
  return undefined;
}

/**
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */
export enum Type {
  STRING,
  NUMBER,
  ARRAY_NUMBER,
  ARRAY_STRING,
  BOOLEAN,
  OBJECT
}

export enum Problem {
  CLASSIFICATION,
  REGRESSION
}

export let problems = {
  "classification": Problem.CLASSIFICATION,
  "regression": Problem.REGRESSION
};

export interface Property {
  name: string;
  type: Type;
  keyMap?: { [key: string]: any };
};

// Add the GUI state.
export class State {

  private static PROPS: Property[] = [
    { name: "d_embed", type: Type.NUMBER },
    { name: "currentFrameIdx", type: Type.NUMBER },
    { name: "experiment", type: Type.OBJECT, keyMap: experiments },
    { name: "networkShape", type: Type.ARRAY_NUMBER },
    { name: "numTransformerBlocks", type: Type.NUMBER },
    { name: "hoverToken", type: Type.NUMBER },

    { name: "batchSize", type: Type.NUMBER },
    { name: "noise", type: Type.NUMBER },
    { name: "seed", type: Type.STRING },
    { name: "problem", type: Type.OBJECT, keyMap: problems },
  ];

  [key: string]: any;
  dEmbed = 2;
  currentFrameIdx: number = 0
  currentFrame = {
    epoch: 0,
    lossTest: 1,
    lossTrain: 1,
    heads: new Array(4).fill(0).map(_ => new Array(10).fill(0).map(_ => new Array(10).fill(0)))
  }
  token_count = 5;
  nodeState: { [id: string]: boolean } = {};
  inputIds: string[] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
  experiment: string = "Temp";
  numTransformerBlocks = 1;
  hoverToken = 0;

  showTestData = false;
  noise = 0;
  batchSize = 10;
  tutorial: string = null;
  percTrainData = 50;
  problem = Problem.CLASSIFICATION;
  initZero = false;
  collectStats = false;
  hiddenLayerControls: any[] = [];
  networkShape: number[] = [4];
  seed: string;

  /**
   * Deserializes the state from the url hash.
   */
  static deserializeState(): State {
    let map: { [key: string]: string } = {};
    for (let keyvalue of window.location.hash.slice(1).split("&")) {
      let [name, value] = keyvalue.split("=");
      map[name] = value;
    }
    let state = new State();

    function hasKey(name: string): boolean {
      return name in map && map[name] != null && map[name].trim() !== "";
    }

    function parseArray(value: string): string[] {
      return value.trim() === "" ? [] : value.split(",");
    }

    // Deserialize regular properties.
    State.PROPS.forEach(({ name, type, keyMap }) => {
      switch (type) {
        case Type.OBJECT:
          if (keyMap == null) {
            throw Error("A key-value map must be provided for state " +
              "variables of type Object");
          }
          if (hasKey(name) && map[name] in keyMap) {
            state[name] = keyMap[map[name]];
          }
          break;
        case Type.NUMBER:
          if (hasKey(name)) {
            // The + operator is for converting a string to a number.
            state[name] = +map[name];
          }
          break;
        case Type.STRING:
          if (hasKey(name)) {
            state[name] = map[name];
          }
          break;
        case Type.BOOLEAN:
          if (hasKey(name)) {
            state[name] = (map[name] === "false" ? false : true);
          }
          break;
        case Type.ARRAY_NUMBER:
          if (name in map) {
            state[name] = parseArray(map[name]).map(Number);
          }
          break;
        case Type.ARRAY_STRING:
          if (name in map) {
            state[name] = parseArray(map[name]);
          }
          break;
        default:
          throw Error("Encountered an unknown type for a state variable");
      }
    });

    // Deserialize state properties that correspond to hiding UI controls.
    state.numTransformerBlocks = state.networkShape.length;
    if (state.seed == null) {
      state.seed = Math.random().toFixed(5);
    }
    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Serialize regular properties.
    let props: string[] = [];
    State.PROPS.forEach(({ name, type, keyMap }) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
      }
      if (type === Type.OBJECT) {
        value = getKeyFromValue(keyMap, value);
      } else if (type === Type.ARRAY_NUMBER ||
        type === Type.ARRAY_STRING) {
        value = value.join(",");
      }
      props.push(`${name}=${value}`);
    });
    // Serialize properties that correspond to hiding UI controls.
    window.location.hash = props.join("&");
  }
}
