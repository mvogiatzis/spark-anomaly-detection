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
    val trainingVec: RDD[Vector] = FeaturesParser.parseFeatures(rawdata)
    val cvLabeledVec: RDD[LabeledPoint] = FeaturesParser.parseFeaturesWithLabel(cvData)

    val data = trainingVec.cache()
    val anDet: AnomalyDetection = new AnomalyDetection()
    //derive model
    val model = anDet.run(data)

    val dataCvVec = cvLabeledVec.cache()
    val optimalModel = anDet.optimize(dataCvVec, model)

    //find outliers in CV
    val cvVec = cvLabeledVec.map(_.features)
    val results = optimalModel.predict(cvVec)
    val outliers = results.filter(_._2).collect()
    outliers.foreach(v => println(v._1))
    println("\nFound %s outliers\n".format(outliers.size))
  }

}