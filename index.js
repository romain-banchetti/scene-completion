const cv = require('opencv4nodejs');
const Jimp = require('jimp');
const GistDescriptor = require('./GistDescriptor');
const Utils = require('./Utils');
const Consts = require('./Consts');

function getSortedMatching (inputImage, availableImages) {
  // On charge l'image que l'on souhaite compléter
  const inputMat = Utils.jimpImageToGrayMat(inputImage.image);
  const inputImageGist = GistDescriptor.compute(inputMat);
  cv.imwrite(`./output/gist/input.png`, inputImageGist);

  // On calcule pour chaque image son gist descriptor
  availableImages = availableImages.map(image => {
    const mat = Utils.jimpImageToGrayMat(image.image);
    const gist = GistDescriptor.compute(mat);
    cv.imwrite(`./output/gist/${image.filename}.png`, gist);
    return {...image, gist};
  });

  // On calcule les distances de chaque images à l'image en entrée (par rapport à leur gist descriptor)
  const sortedImages = availableImages
    .map(image => ({...image, distance: Utils.matGrayDistance(inputImageGist, image.gist)}))
    .sort((a, b) => a.distance < b.distance ? -1 : 1);

  return sortedImages;
}

async function main () {
  // Image choisie sur laquelle un trou sera ajouté et sur laquelle l'algorithme sera utilisé
  const inputImageFilename = '20';

  // Nombre d'images disponibles dans le dossier input
  const availableImageCount = 26;


  /**
   * On charge toutes les images nécessaires
   */
  // On charge l'image en entrée
  let inputJimpImage = await Jimp.read(`./input/${inputImageFilename}.jpg`);
  inputJimpImage = inputJimpImage.cover(Consts.WorkingImageSize, Consts.WorkingImageSize);

  // On charge les images disponibles dans un tableau
  const availableImages = [];
  for (let i = 1; i <= availableImageCount; i++) {
    const imageFilename = `${i}`;

    if (imageFilename === inputImageFilename) {
      continue;
    }

    let image = await Jimp.read(`./input/${imageFilename}.jpg`);
    image = image.cover(Consts.WorkingImageSize, Consts.WorkingImageSize);

    availableImages.push({filename: imageFilename, image});
  }


  /**
   * On trouve les 5 images les plus proches de l'image en entrée
   */
  // On classes les images disponibles par leur distance avec l'image en entrée
  const matchingResults = getSortedMatching({filename: inputImageFilename, image: inputJimpImage}, availableImages);

  // On ne garde que les "NClosestMatching" images les plus proches
  const bestMatchingImages = matchingResults.slice(0, Consts.NClosestMatching);
  bestMatchingImages.forEach((matching, index) => matching.image.write(`./output/matching_${index + 1}.png`));
  console.log('Images les plus proches :', bestMatchingImages.map(i => ({filename: i.filename, distance: i.distance})));
  // return;

  /**
   * On crée l'image avec le trou, et on récupère le "local context"
   */
  // On retire un carré de l'image en entrée (pour simuler le trou)
  const inputImageWithHole = inputJimpImage.clone().composite(new Jimp(Consts.HiddenRegionWidth, Consts.HiddenRegionHeight, '#000000'), Consts.HiddenRegionX, Consts.HiddenRegionY);
  inputImageWithHole.write('./output/input_with_hole.png');

  // On récupère le contour de la zone retirée (une marge de chaque côté du carré)
  const inputLocalContextImage = inputImageWithHole.clone().crop(Consts.HiddenRegionX - Consts.LocalContextMarging, Consts.HiddenRegionY - Consts.LocalContextMarging, Consts.HiddenRegionWidth + Consts.LocalContextMarging * 2, Consts.HiddenRegionHeight + Consts.LocalContextMarging * 2);
  inputLocalContextImage.write('./output/input_local_context.png');
  const inputLocalContextMat = Utils.jimpImageToMat(inputLocalContextImage);


  /**
   * On cherche la zone qui est la plus proche de la zone recherchée dans chaque matching image
   */
  let minDistance = null;
  let bestMatchingImage = null;
  let bestMatchingLocalContextPosition = null;
  let bestMatchingLocalContext = null;

  for (const matchingImage of bestMatchingImages) {
    for (let i = 0; i < matchingImage.image.bitmap.width - (Consts.HiddenRegionWidth + 2 * Consts.LocalContextMarging); i++) {
      for (let j = 0; j < matchingImage.image.bitmap.height - (Consts.HiddenRegionHeight + 2 * Consts.LocalContextMarging); j++) {
        const matchingLocalContextImage = matchingImage.image.clone().crop(i, j, Consts.HiddenRegionWidth + 2 * Consts.LocalContextMarging, Consts.HiddenRegionHeight + 2 * Consts.LocalContextMarging);
        const distance = Utils.matDistanceAsL_a_b(inputLocalContextMat, Utils.jimpImageToMat(matchingLocalContextImage), true);

        if (bestMatchingImage === null || distance < minDistance) {
          minDistance = distance;
          bestMatchingImage = matchingImage;
          bestMatchingLocalContextPosition = {x: i, y: j};
          bestMatchingLocalContext = matchingLocalContextImage;
        }
      }
    }
  }

  bestMatchingLocalContext.write('./output/best_matching_local_context.png');
  console.log('Meilleur zone trouvée :', bestMatchingImage.filename, bestMatchingLocalContextPosition, minDistance);


  /**
   * On colle la meilleure zone trouvée à l'emplacement du trou dans l'image en entrée
   */
  // Sans algorithme "seamless"
  const resultImageWithoutSeamless = inputImageWithHole.clone().composite(bestMatchingLocalContext.clone().crop(Consts.LocalContextMarging, Consts.LocalContextMarging, Consts.HiddenRegionWidth, Consts.HiddenRegionHeight), Consts.HiddenRegionX, Consts.HiddenRegionY);
  resultImageWithoutSeamless.write('./output/result_without_seamless.png');

  // Avec algorithme "seamless"
  const mask = new cv.Mat(bestMatchingLocalContext.bitmap.width, bestMatchingLocalContext.bitmap.width, cv.CV_8UC3, [255, 255, 255]);
  const resultMatWithSeamless = cv.seamlessClone(Utils.jimpImageToMat(bestMatchingLocalContext), Utils.jimpImageToMat(inputImageWithHole), mask, new cv.Point(Consts.HiddenRegionX + Consts.HiddenRegionWidth / 2, Consts.HiddenRegionY + Consts.HiddenRegionHeight / 2), cv.NORMAL_CLONE);
  Utils.matToJimpImage(resultMatWithSeamless).write('./output/result_with_seamless.png');
}

main();
