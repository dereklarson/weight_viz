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
 * The data type of a state variable. Used for determining the
 * (de)serialization method.
 */


export enum Type {
  BOOLEAN,
  NUMBER,
  STRING,
  ARRAY_NUMBER,
  ARRAY_STRING,
}

export interface Property {
  name: string;
  type: Type;
};

export class State {

  /** All PROPS are (de)serialized via the URL */
  private static PROPS: Property[] = [
    { name: "experiment", type: Type.STRING },
    { name: "currentTag", type: Type.STRING },
    { name: "currentTab", type: Type.STRING },
    // { name: "currentFrameIdx", type: Type.NUMBER },
    { name: "useContext", type: Type.BOOLEAN },
  ];

  experiment: string = "parity";
  currentTag: string = "de@3_nh@1";
  currentTab: string = "model";
  currentFrameIdx: number = 0;
  useContext: boolean = false;

  // Should we serialize the following?
  selectedTokenId: string = null;
  selectedNodeId: string = null;
  context: number[] = [0, 1];

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
    State.PROPS.forEach(({ name, type }) => {
      switch (type) {
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
    return state;
  }

  /**
   * Serializes the state into the url hash.
   */
  serialize() {
    // Serialize regular properties.
    let props: string[] = [];
    State.PROPS.forEach(({ name, type }) => {
      let value = this[name];
      // Don't serialize missing values.
      if (value == null) {
        return;
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
