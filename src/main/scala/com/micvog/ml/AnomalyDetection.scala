package com.micvog.ml

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD

/**
  * Anomaly Detection algorithm
  */
class AnomalyDetection extends Serializable with Logging {

  val default_epsilon: Double = 0.1

  def run(data: RDD[Vector]): AnomalyDetectionModel = {
    val sc = data.sparkContext

    val stats: MultivariateStatisticalSummary = Statistics.colStats(data)
    val mean: Vector = stats.mean
//    val mean = Vectors.dense(Array(14.1122257839456, 14.9977105081362))
    val variances: Vector = stats.variance
//    val variances: Vector = Vectors.dense(Array(1.83263141349452, 1.70974533082878))
    logInfo("MEAN %s VARIANCE %s".format(mean, variances))

    new AnomalyDetectionModel(mean, variances, default_epsilon)
  }

  /**
    * Uses the labeled input points to optimize the epsilon parameter by finding the best F1 Score
    * @param crossValData
    * @param anomalyDetectionModel
    * @return
    */
  def optimize(crossValData: RDD[LabeledPoint], anomalyDetectionModel: AnomalyDetectionModel) = {
    val sc = crossValData.sparkContext
    val bcMean = sc.broadcast(anomalyDetectionModel.means)
    val bcVar = sc.broadcast(anomalyDetectionModel.variances)

    //compute probability density function for each example in the cross validation set
    val probsCV: RDD[Double] = crossValData.map(labeledpoint =>
      AnomalyDetection.probFunction(labeledpoint.features, bcMean.value, bcVar.value)
    )

    //select epsilon
    crossValData.persist()
    val epsilonWithF1Score: (Double, Double) = evaluate(crossValData, probsCV)
    crossValData.unpersist()

    logInfo("Best epsilon %s F1 score %s".format(epsilonWithF1Score._1, epsilonWithF1Score._2))
    new AnomalyDetectionModel(anomalyDetectionModel.means, anomalyDetectionModel.variances, epsilonWithF1Score._1)
  }

  /**
    *  Finds the best threshold to use for selecting outliers based on the results from a validation set and the ground truth.
    *
    * @param crossValData labeled data
    * @param probsCV probability density function as calculated for the labeled data
    * @return Epsilon and the F1 score
    */
  private def evaluate(crossValData: RDD[LabeledPoint], probsCV: RDD[Double]) = {

    val minPval: Double = probsCV.min()
    val maxPval: Double = probsCV.max()
    logInfo("minPVal: %s, maxPVal %s".format(minPval, maxPval))
    val sc = probsCV.sparkContext

    var bestEpsilon = 0D
    var bestF1 = 0D

    //starting from epsilon to
    val stepsize = (maxPval - minPval) / 1000.0

    for (epsilon <- minPval to maxPval by stepsize){

      val bcepsilon = sc.broadcast(epsilon)

      val ourPredictions: RDD[Double] = probsCV.map{ prob =>
        if (prob < bcepsilon.value)
          1.0 //anomaly
        else
          0.0
      }
      val labelAndPredictions: RDD[(Double, Double)] = crossValData.map(_.label).zip(ourPredictions)
      val labelWithPredictionCached: RDD[(Double, Double)] = labelAndPredictions

      val falsePositives = countStatisticalMeasure(labelWithPredictionCached, 0.0, 1.0)
      val truePositives = countStatisticalMeasure(labelWithPredictionCached, 1.0, 1.0)
      val falseNegatives = countStatisticalMeasure(labelWithPredictionCached, 1.0, 0.0)

      val precision = truePositives / Math.max(1.0, truePositives + falsePositives)
      val recall = truePositives / Math.max(1.0, truePositives + falseNegatives)

      val f1Score = 2.0 * precision * recall / (precision + recall)

      if (f1Score > bestF1){
        bestF1 = f1Score
        bestEpsilon = epsilon
      }
    }

    (bestEpsilon, bestF1)
  }

  /**
    * Function to calculate true / false positives, negatives
    *
    * @param labelWithPredictionCached
    * @param labelVal
    * @param predictionVal
    * @return
    */
  private def countStatisticalMeasure(labelWithPredictionCached: RDD[(Double, Double)], labelVal: Double, predictionVal: Double): Double = {
    labelWithPredictionCached.filter { labelWithPrediction =>
      val label = labelWithPrediction._1
      val prediction = labelWithPrediction._2
      label == labelVal && prediction == predictionVal
    }.count().toDouble
  }

}



object AnomalyDetection {


  /**
    * True if the given point is an anomaly, false otherwise
    * @param point
    * @param means
    * @param variances
    * @param epsilon
    * @return
    */
  private[ml] def predict (point: Vector, means: Vector, variances: Vector, epsilon: Double): Boolean = {
    probFunction(point, means, variances) < epsilon
  }

  private[ml] def probFunction(point: Vector, means: Vector, variances: Vector): Double = {
    val tripletByFeature: List[(Double, Double, Double)] = (point.toArray, means.toArray, variances.toArray).zipped.toList
    tripletByFeature.map { triplet =>
      val x = triplet._1
      val mean = triplet._2
      val variance = triplet._3
      val expValue = Math.pow(Math.E, -0.5 * Math.pow(x - mean,2) / variance)
      (1.0 / (Math.sqrt(variance) * Math.sqrt(2.0 * Math.PI))) * expValue
    }.product
  }





}



