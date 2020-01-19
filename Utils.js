const cv = require('opencv4nodejs');
const Jimp = require('jimp');

class Utils
{
  static grayMatToJimpImage (mat) {
    mat = mat.cvtColor(cv.COLOR_GRAY2RGBA);
    return new Jimp({
      width: mat.cols,
      height: mat.rows,
      data: mat.getData()
    })
  }

  static matToJimpImage (mat) {
    mat = mat.cvtColor(cv.COLOR_BGR2RGBA);
    return new Jimp({
      width: mat.cols,
      height: mat.rows,
      data: mat.getData()
    })
  }

  static jimpImageToGrayMat (image) {
    let mat = new cv.Mat(image.bitmap.data, image.bitmap.width, image.bitmap.height, cv.CV_8UC4);
    mat = mat.cvtColor(cv.COLOR_RGBA2GRAY);
    return mat;
  }

  static jimpImageToMat (image) {
    let mat = new cv.Mat(image.bitmap.data, image.bitmap.width, image.bitmap.height, cv.CV_8UC4);
    mat = mat.cvtColor(cv.COLOR_RGBA2BGR);
    return mat;
  }

  static concatJimpImages (images) {
    const resultImage = new Jimp(images.length * images[0].bitmap.width, images[0].bitmap.height, '#000000');

    for (let i = 0; i < images.length; i++) {
      resultImage.composite(images[i], i * images[0].bitmap.width, 0);
    }

    return resultImage;
  }

  static matGrayDistance (mat1, mat2, ignoreNullPixels = false) {
    let sum = 0;
    for (let i = 0; i < mat1.rows; i++) {
      for (let j = 0; j < mat1.cols; j++) {
        const p1 = mat1.at(i, j);
        const p2 = mat2.at(i, j);
        if (ignoreNullPixels && (!p1 || !p2)) {
          continue;
        }
        sum += Math.pow(p1 - p2, 2)
      }
    }
    return Math.sqrt(sum);
  }

  static matDistanceAsL_a_b (mat1, mat2, ignoreNullPixels = false) {
    let sum = 0;
    for (let i = 0; i < mat1.rows; i++) {
      for (let j = 0; j < mat1.cols; j++) {
        const p1 = mat1.at(i, j);
        const p2 = mat2.at(i, j);
        if (ignoreNullPixels && (!p1.x && !p1.y && !p1.z || !p2.x && !p2.y && !p2.z)) {
          continue;
        }
        sum += Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2)
      }
    }
    return Math.sqrt(sum);
  }
}

module.exports = Utils;