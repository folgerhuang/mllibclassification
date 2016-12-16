import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionModel, LinearRegressionWithSGD}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by root on 2016/8/17 0021.
  */

object LinearRegressionWithSGD_new {
  //  F:\\pensionRisk\data\LinearRWithSGD\train F:\\pensionRisk\data\LinearRWithSGD\model 30 0.85 local
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    if (args.length < 5) {
      System.err.println("Usage: LinearRWithSGD <inputPath> <modelPath> <iterNum> <step> <master> [<AppName>]")
      System.err.println("eg: hdfs://192.168.57.104:8020/user/000000_0 10 0.1 spark://192.168.57.104:7077 LinearRWithSGD")
      System.exit(1)
    }
    val appName = if(args.length>5) args(5) else "LinearRWithSGD"
    val conf = new SparkConf().setAppName(appName).setMaster(args(4))
    val sc = new SparkContext(conf)
    val traindata: RDD[LabeledPoint] = MLUtils.loadLabeledPoints(sc,args(0))
    val splitRdd: Array[RDD[LabeledPoint]] = traindata.randomSplit(Array(1.0,9.0))
    val testData: RDD[LabeledPoint] = splitRdd(0)
    val realTrainData = splitRdd(1)
    val model: LinearRegressionModel = LinearRegressionWithSGD.train(realTrainData, args(2).toInt, args(3).toDouble)
    val predict: RDD[Double] = model.predict(testData.map(_.features))
    val zip: RDD[(Double, Double)] = predict.zip(testData.map(_.label))
    zip.foreach(x=>println(x._1+"\t\t"+x._2))

    val loss = zip.map(x=>{
      val a = x._1 - x._2
      a*a
    }).mean()
    println("loss======"+loss)
    model.save(sc,args(1))
  }
}
