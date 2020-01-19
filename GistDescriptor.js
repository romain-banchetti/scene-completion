const cv = require('opencv4nodejs');

class GistDescriptor
{
  static compute (src) {
    const orientationCount = 6;
    const scaleCount = 5;
    const regionWidth = src.cols / 4;
    const regionHeight = src.rows / 4;

    // On initialise l'image finale
    const gist = new cv.Mat(orientationCount * 4, scaleCount * 4, cv.CV_8UC1, 0);

    // On génère les 6*5 = 30 images à partir de filtres de Gabor
    // const images = [];
    for (let i = 0; i < orientationCount; i++) {
      for (let j = 0; j < 5; j++) {
        const mat = GistDescriptor.applyGaborFilterToGrayMat(src, i * Math.PI / 6, j + 2);
        // images.push(cvGrayMapToJimpImage(mat));

        // Pour chaque image issue d'un filtre de Gabor
        for (let x = 0; x < 4; x++) {
          for (let y = 0; y < 4; y++) {
            // On découpe en 16 zones
            const region = mat.getRegion(new cv.Rect(x * regionWidth, y * regionHeight, regionWidth, regionHeight));
            // On calcule la moyenne de chaque zone qui correspond chacune à un pixel du gist final
            const mean = Math.round(region.mean().w);
            gist.set(i * 4 + x, j * 4 + y, mean);
          }
        }
      }
    }

    return gist;
  }

  /**
   * Renvoie un noyau de Gabor
   */
  static getGaborKernel (ksize, sigma, theta, lambd, gamma, psi = Math.PI *0.5, ktype = cv.CV_64F) {
    const sigma_x = sigma;
    const sigma_y = sigma / gamma;
    const nstds = 3;
    let xmin, xmax, ymin, ymax;
    const c = Math.cos(theta);
    const s = Math.sin(theta);

    if( ksize.width > 0 )
      xmax = ksize.width / 2;
    else
      xmax = cv.cvRound(Math.max(Math.abs(nstds * sigma_x * c), Math.abs(nstds * sigma_y * s)));

    if( ksize.height > 0 )
      ymax = ksize.height / 2;
    else
      ymax = cv.cvRound(Math.max(Math.abs(nstds * sigma_x * s), Math.abs(nstds * sigma_y * c)));

    xmin = -xmax;
    ymin = -ymax;

    const kernel = new cv.Mat(ymax - ymin + 1, xmax - xmin + 1, ktype);
    const scale = 1;
    const ex = -0.5 / (sigma_x * sigma_x);
    const ey = -0.5 / (sigma_y * sigma_y);
    const cscale = Math.PI * 2 / lambd;

    for( let y = ymin; y <= ymax; y++ ) {
      for( let x = xmin; x <= xmax; x++ ) {
        const xr = x * c + y * s;
        const yr = -x * s + y * c;

        let v = scale*Math.exp(ex * xr * xr + ey * yr * yr) * Math.cos(cscale * xr + psi);
        kernel.set(ymax - y, xmax - x, v);
      }
    }

    return kernel;
  }

  static getGaborKernelFromOrientationAndScale (orientation, scale = 2) {
    return GistDescriptor.getGaborKernel(new cv.Size(30, 30), 2, orientation, scale, 0.5, 0, cv.CV_32F);
  }

  static applyGaborFilterToGrayMat (src, orientation, scale) {
    src = src.convertTo(cv.CV_32F);
    let kernel = GistDescriptor.getGaborKernelFromOrientationAndScale(orientation, scale);

    // // Décommenter pour voir le noyau
    // kernel = kernel.normalize(255, 0, cv.NORM_MINMAX);
    // kernel = kernel.convertTo(cv.CV_8U);
    // return kernel;

    let dst = src.filter2D(cv.CV_32F, kernel);
    dst = dst.normalize(255, 0, cv.NORM_MINMAX);
    dst = dst.convertTo(cv.CV_8U);
    return dst;
  }
}

module.exports = GistDescriptor;