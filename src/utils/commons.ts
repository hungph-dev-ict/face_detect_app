import * as path from 'path';
import * as fs from 'fs';

export const dataPath = path.resolve(__dirname, '../../data');
export const modelFilePath = path.resolve(__dirname, '../../data/model');

export function getDataPath() {
  return dataPath;
}

export function getModelFilePath() {
  return modelFilePath;
}

export function ensureModelFilePathDirExists() {
  if (!fs.existsSync(modelFilePath)) {
    fs.mkdirSync(modelFilePath);
  }
}
