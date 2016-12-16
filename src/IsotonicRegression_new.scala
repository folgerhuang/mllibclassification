import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.IsotonicRegression

/**
  * Created by root on 2016/8/16 0020.
  */
object IsotonicRegression_new {
//  F:\额外项目\pensionRisk\data\IsR\train\sample_isotonic_regression_data.txt F:\额外项目\pensionRisk\data\IsR\model true local
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    if (args.length < 4) {
      System.err.println("Usage: LRwithLGD <inputPath> <modelPath> Isotonic <master> [<AppName>]")
      System.err.println("eg: hdfs://192.168.57.104:8020/user/000000_0 hdfs://192.168.57.104:8020/user/model true  spark://192.168.57.104:7077  IsotonicRegression")
      System.exit(1)
    }
    val appName = if(args.length>4) args(4) else "IsotonicRegression"
  val conf = new SparkConf().setAppName(appName).setMaster(args(3))
  val sc = new SparkContext(conf)
    var isotonic =true
    isotonic = args(2) match {
      case "true" => true
      case "false" => false
     }
  val data = sc.textFile(args(0))
  val parsedData = data.map { line =>
    val parts = line.split(',').map(_.toDouble)
    (parts(0), parts(1), 1.0)
  }

  val splitRdd = parsedData.randomSplit(Array(1.0,9.0))
  val testData = splitRdd(0)
  val realTrainData = splitRdd(1)

  val model = new IsotonicRegression().setIsotonic(isotonic).run(realTrainData)
  val predictionAndLabel = testData.map { point =>
    val predictedLabel = model.predict(point._2)
    (predictedLabel, point._1)
  }
  val meanSquaredError = predictionAndLabel.map { case p => math.pow((p._1 - p._2), 2) }.mean()
  println("meanSquaredError = " + meanSquaredError)
  model.save(sc,args(1))

  }
}
