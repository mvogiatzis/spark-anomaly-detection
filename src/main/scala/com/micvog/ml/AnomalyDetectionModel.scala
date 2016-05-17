package com.micvog.ml

import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.RDD

class AnomalyDetectionModel  (
                               val means: Vector,
                               val variances: Vector,
                               val epsilon: Double)
{
  /**
    *
    * @param point
    * @return
    */
  def predict(point: Vector): Boolean = {
    AnomalyDetection.predict(point, means, variances, epsilon)
  }

  def predict(points: RDD[Vector]): RDD[Boolean] = {
    points.map(p => AnomalyDetection.predict(p, means, variances, epsilon))
  }

}