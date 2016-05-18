package com.micvog.ml

import org.apache.spark.mllib.linalg.Vectors
import org.scalatest.{FlatSpec, FunSuite, Matchers}

class AnomalyDetection$Test extends FlatSpec with Matchers {

  val point = Vectors.dense(Array(14.8593411857427, 14.9006647394062))
  val means = Vectors.dense(Array(14.1122257839456, 14.9977105081362))
  val variances = Vectors.dense(Array(1.83263141349452, 1.70974533082878))

  "probFunction" should "return correct product value" in {
    val p = AnomalyDetection.probFunction(point, means, variances)
    assert(p === 0.0769984879544  +- 0.0001)
  }

  "predict" should "predict the anomaly" in {
    assert(!AnomalyDetection.predict(point, means, variances, 0.05))
  }

  "predict" should "predict non anomaly" in {
    assert(AnomalyDetection.predict(point, means, variances, 0.08))
  }

}
