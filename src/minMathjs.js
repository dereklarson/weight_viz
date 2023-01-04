/** This file enables Webpack to create a smaller bundle by identifying only
 * the needed portions of MathJS and lets tree-shaking remove the rest.
 * TypeScript versions of this didn't seem to work, thus it remains a 'js' file.
 */

import {
    addDependencies,
    cloneDependencies,
    create,
    matrixDependencies,
    multiplyDependencies,
    transposeDependencies
} from 'mathjs';

const config = {}

export const { add, clone, matrix, multiply, transpose } = create({
    addDependencies,
    cloneDependencies,
    matrixDependencies,
    multiplyDependencies,
    transposeDependencies
}, config)