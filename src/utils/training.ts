import * as path from "path";
import * as fs from "fs";
import * as fr from "face-recognition";
import {
  getDataPath,
  getModelFilePath,
  ensureModelFilePathDirExists
} from "./commons";

export async function trainingNewModel(nameClass: Array<string>) {
  fr.winKillProcessOnExit();
  ensureModelFilePathDirExists();

  const trainedModelFile = "faceRecognitionModel_HungPH.json";
  const trainedModelFilePath = path.resolve(
    getModelFilePath(),
    trainedModelFile
  );

  const facesPath = path.resolve(getDataPath(), "img/faces");
  const classNames = nameClass;

  const recognizer = fr.FaceRecognizer();

  console.log("Start training recognizer, please waiting ...");
  const allFiles = fs.readdirSync(facesPath);
  const imagesByClass = classNames.map(c =>
    allFiles
      .filter(f => f.includes(c))
      .map(f => path.join(facesPath, f))
      .map(fp => fr.loadImage(fp))
  );

  imagesByClass.forEach((faces, label) =>
    recognizer.addFaces(faces, classNames[label])
  );

  fs.writeFileSync(
    trainedModelFilePath,
    JSON.stringify(recognizer.serialize()),
    {
      encoding: "UTF-8",
      flag: "a+"
    }
  );

  const newModel = fs
    .readFileSync(trainedModelFilePath)
    .toString()
    .replace("}][{", "}, {");

  fs.writeFileSync(trainedModelFilePath, newModel);
}
