import * as cv from "opencv4nodejs";
import * as path from "path";
import { getDataPath } from "./commons";

export function createImageData(
  classImageName: string,
  index: number,
  frame: any
) {
  const facesPath = path.resolve(getDataPath(), "img/faces");
  const facesFileName = path.resolve(
    facesPath,
    classImageName + "_" + index + ".png"
  );
  cv.imwrite(facesFileName, frame);
}
