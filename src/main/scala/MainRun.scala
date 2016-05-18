/* SimpleApp.scala */
import com.micvog.ml.{AnomalyDetection, FeaturesParser}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

object MainRun {

  val rawFilePath = "./src/test/resources/training.csv"
  val cvFilePath = "./src/test/resources/cross_val.csv"

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("Anomaly Detection Spark")
    val sc = new SparkContext(conf)

    val rawdata = sc.textFile(rawFilePath, 2).cache()
    val cvData = sc.textFile(cvFilePath, 2).cache()

    //convert raw data to vectors
    val vectors: RDD[Vector] = FeaturesParser.parseFeatures(rawdata)
    val cvVectors: RDD[LabeledPoint] = FeaturesParser.parseFeaturesWithLabel(cvData)

    val data = vectors.cache()
    val anDet: AnomalyDetection = new AnomalyDetection()
    //derive model
    val model = anDet.run(data)

    val dataCvVec = cvVectors.cache()
    val optimalModel = anDet.optimize(dataCvVec, model)
  }

}