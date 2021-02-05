const PNN = (function() {
 
  // https://scialert.net/fulltext/?doi=jai.2011.288.294
  // https://www.cse.unr.edu/~looney/cs773b/PNNtutorial.pdf
  // https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826#:~:text=To%20calculate%20accuracy%2C%20use%20the,or%20(1%2DAccuracy).
  
  let featuresCount;
  let k = 3;
  let kValues = [0,1,2];
  let kExemplar = [33,33,34];
  
  let hidden = [];
  let output = [];
  let min = [];
  let max = [];
  
  function getMin(dataset, featureIndex) {
    let min = dataset[0][featureIndex];
    for (let i=1; i<dataset.length; i++) {
      min = Math.min(min, dataset[i][featureIndex]);
    }
    return min;
  }
  
  function getMax(dataset, featureIndex) {
    let max = dataset[0][featureIndex];
    for (let i=1; i<dataset.length; i++) {
      max = Math.max(max, dataset[i][featureIndex]);
    }
    return max;
  }

  // min-max normalization
  function normalize(dataset) {
    for (let i=0; i<featuresCount; i++) {
      min[i] = getMin(dataset, i);
      max[i] = getMax(dataset, i);
      for (let data of dataset) {
        data[i] = (data[i]-min[i])/(max[i]-min[i]);
      }
    }
  }
  
  function normalizeInput(data) {
    for (let i=0; i<featuresCount; i++) {
      data[i] = (data[i]-min[i])/(max[i]-min[i]);
    }
    return data;
  }

  function initLayers(trainingSet, trainingY) {
    
    while (output.length < k) {
      let i = output.length;
      let neuron = {
        kExemplar: kExemplar[i],
        smoothingParam: calculateSmoothingParam(kValues[i], kExemplar[i]),
        sumDistance: 0,
        sumInput: sumInput,
        computeGaussianSum: computeGaussianSum,
        sourceNode: [],
        kIndex: i,
      };
      output.push(neuron);
    }
    
    while (hidden.length < trainingSet.length) {
      let i = hidden.length;
      let neuron = {
        weight: trainingSet[i],
        computeDistance: computeDistance,
        distance: 0,
      };
      output[kValues.indexOf(trainingY[i])].sourceNode.push(neuron);
      hidden.push(neuron);
    }
    
    L('hidden layer', hidden);
    L('summation layer', output);
    
    function calculateSumDistance(i,j) {
      let sum = 0;
      for (let k=0; k<featuresCount; k++) {
        sum += Math.pow(trainingSet[i][k]-trainingSet[j][k],2);
      }
      return sum;
    }
    
    function calculateSmoothingParam(kValue, kExemplar) {
      return 0.3;
      let dSum = 0;
      for (let i=0; i<trainingSet.length; i++) {
        if (trainingY[i] != kValue)
          continue;
        
        let dMin = null;
        for (let j=0; j<trainingSet.length; j++) {
          if (i == j)
            continue;
            
          let distance = Math.sqrt( calculateSumDistance(i, j) );
          if (dMin === null) {
            dMin = distance;
          } else {
            dMin = Math.min(dMin, distance);
          }
        }
        dSum += dMin;
      }
      
      let g = 0.5;
      L('---',g*dSum/kExemplar);
      return g*dSum/kExemplar;
    }
  }
  
  function computeGaussianSum() {
    let sum = 0;
    for (let node of this.sourceNode) {
      sum += Math.exp(-1*Math.pow(node.distance,2)/(2*Math.pow(this.smoothingParam,2)));
    }
    return sum;
  }
  
  function computeDistance(input) {
    let sum = 0;
    for (let i=0; i<featuresCount; i++) {
      sum += Math.pow(input[i] - this.weight[i], 2);
    }
    this.distance = Math.sqrt(sum);
  }
  
  function sumInput(m) {
    const PI = 3.14;
    let result = 1 / ( Math.pow( 2*PI, m/2) * Math.pow(this.smoothingParam, m) * this.kExemplar) * this.computeGaussianSum();
    return result;
  }
  
  function feedInput(input) {
    
    for (let neuron of hidden) {
      neuron.computeDistance(input);
    }

    let maxSum = null;
    let choosenK = null;
    
    for (let neuron of output) {
      let sum = neuron.sumInput(input.length);
      if (maxSum === null) {
        maxSum = sum;
        choosenK = kValues[neuron.kIndex];
      } else {
        if (sum > maxSum) {
          maxSum = sum;
          choosenK = kValues[neuron.kIndex];
        }
      }
    }
    
    // L('Choosen class : ' + choosenK);
    
    return choosenK;
  }
  
  function train(json_object) {
    let dataset = [];
    let targetClass = [];
    let classList = ['Iris-setosa','Iris-versicolor','Iris-virginica'];
    
    for (let row of json_object) {
      dataset.push([
        Number(row.SepalLengthCm),
        Number(row.SepalWidthCm),
        Number(row.PetalLengthCm),
        Number(row.PetalWidthCm),
      ]);
      targetClass.push(classList.indexOf(row.Species));
    }

    featuresCount = dataset[0].length;
    
    normalize(dataset);
    initLayers(dataset, targetClass);
  }
  
  function test(json_object) {
    
    let dataset = [];
    let targetClass = [];
    let classList = ['Iris-setosa','Iris-versicolor','Iris-virginica'];
    
    for (let row of json_object) {
      dataset.push([
        Number(row.SepalLengthCm),
        Number(row.SepalWidthCm),
        Number(row.PetalLengthCm),
        Number(row.PetalWidthCm),
        classList.indexOf(row.Species),
      ])
    };
    
    let confusionMatrix = [[0,0,0],[0,0,0],[0,0,0]];
    
    for (let i=0; i<dataset.length; i++) {
      let classifiedClass = feedInput(normalizeInput(dataset[i]));
      confusionMatrix[dataset[i][featuresCount]][classifiedClass] += 1;
    }
    
    calculateClassificationResult(confusionMatrix);
  }
  
  function calculateClassificationResult(confusionMatrix) {
    L('--- confusion matrix');
    L(confusionMatrix)
    // L(confusionMatrix[0])
    // L(confusionMatrix[1])
    // L(confusionMatrix[2])
    
    let TP = [];
    let TN = [];
    let FP = [];
    let FN = [];
    let P = [];
    let N = [];
    let accuracy = [];
    let precision = [];
    let recall = [];
    let f1 = [];
    
    let matrixSize = confusionMatrix.length;
    
    for (let i=0; i<matrixSize; i++) {
      TP[i] = confusionMatrix[i][i];
    }
    L('--- TP', TP);
    
    for (let i=0; i<matrixSize; i++) {
      let sum = 0;
      for (let j=0; j<matrixSize; j++) {
        if (j == i)
          continue
        for (let k=0; k<matrixSize; k++) {
          if (k == i)
            continue;
          sum += confusionMatrix[j][k];
        }
      }
      TN[i] = sum;
    }
    L('--- TN', TN);
    
    for (let i=0; i<matrixSize; i++) {
      let sum = 0;
      for (let j=0; j<matrixSize; j++) {
        if (j == i)
          continue
        sum += confusionMatrix[i][j];
      }
      FP[i] = sum;
    }
    L('--- FP', FP);
    
    for (let i=0; i<matrixSize; i++) {
      let sum = 0;
      for (let j=0; j<matrixSize; j++) {
        if (j == i)
          continue
        sum += confusionMatrix[j][i];
      }
      FN[i] = sum;
    }
    L('--- FN', FN);
    
    // P = TP + FN
    for (let i=0; i<matrixSize; i++) {
      P[i] = TP[i] + FN[i];
    }
    L('--- P', P);
    
    // N = FP + TN
    for (let i=0; i<matrixSize; i++) {
      N[i] = FP[i] + TN[i];
    }
    L('--- N', N);
    
    // accuracy
    // (TP + TN) / (P + N)
    for (let i=0; i<matrixSize; i++) {
      accuracy[i] = (TP[i] + TN[i]) / (P[i] + N[i]);
    }
    L('--- Accuracy', accuracy);
    
    // precision
    // TP / (TP + FP)
    for (let i=0; i<matrixSize; i++) {
      precision[i] = TP[i] / (TP[i] + FP[i]);
    }
    L('--- Precision', precision);
    
    // recall
    // TP / (TP + FN)
    for (let i=0; i<matrixSize; i++) {
      recall[i] = TP[i] / (TP[i] + FN[i]);
    }
    L('--- Recall', recall);
    
    // F1
    // (2 * precision * recall) / (precision + recall)
    for (let i=0; i<matrixSize; i++) {
      f1[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]);
    }
    L('--- F1', f1);
    
    let sumTP = TP.reduce((a,b) => a+b);
    let sumTN = TN.reduce((a,b) => a+b);
    let sumFP = FP.reduce((a,b) => a+b);
    let sumFN = FN.reduce((a,b) => a+b);
    let sumP = P.reduce((a,b) => a+b);
    let sumN = N.reduce((a,b) => a+b);
    
    let sumAccuracy = (sumTP + sumTN) / (sumP + sumN);
    L('--- Avg. Accuracy', sumAccuracy);
    
    let sumPrecision = sumTP / (sumTP + sumFP);
    L('--- Avg. Precision', sumPrecision);

    let sumRecall = sumTP / (sumTP + sumFN);
    L('--- Avg. Recall', sumRecall);
    
    let microF1 = (2 * sumPrecision * sumRecall) / (sumPrecision + sumRecall)
    L('--- Micro-F1', microF1);
  }
 
  return {
    train,
    test,
  };
  
})();