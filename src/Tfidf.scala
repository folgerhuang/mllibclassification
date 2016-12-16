import breeze.linalg._
import org.apache.spark.mllib.feature.{IDFModel, HashingTF, IDF}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.{SparseVector => SV}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.io.Source

/**
  * Created by Administrator on 2016/10/25 0025.
  */
object Tfidf {
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("TfIdfTest").setMaster("local")
    val sc = new SparkContext(conf)
//    val a: RDD[Int] = sc.parallelize(Array(1,2,3))
//    val b: RDD[Int] = sc.parallelize(Array(4,5,6))
//    val c: RDD[(Int, Int)] = a.zip(b)
//    c.foreach(println(_))
//    文件里面每行代表一个document，后面zip一个index是为了标示文章
    val documents: RDD[(Seq[String], Long)] = sc.parallelize(Source.fromFile("E:\\课程\\mllib\\mllib第一天\\课程附件资料\\doc.txt").getLines().filter(_.trim.length > 0).toSeq).map(_.split(" ").toSeq).zipWithIndex()

//    构建TF对象，参数就是稀疏大向量的长度，这里不写默认1<<20
    val hashingTF = new HashingTF(Math.pow(2, 18).toInt)
    val tf_num_pairs: RDD[(Long, linalg.Vector)] = documents.map {
      case (seq, num) =>
//        计算每个文档的词频
        val tf: linalg.Vector = hashingTF.transform(seq)
        (num, tf)
    }

    tf_num_pairs.cache()
//    根据所有的词频计算逆词频
    val idf: IDFModel = new IDF().fit(tf_num_pairs.values)
    val num_idf_pairs: RDD[(Long, linalg.Vector)] = tf_num_pairs.mapValues(v => idf.transform(v))
//    求文章两两之间的相似度，需要笛卡尔积，把自己广播一下
    val b_num_idf_pairs = sc.broadcast(num_idf_pairs.collect())



//求document的相似度
//    val docSims = num_idf_pairs.flatMap {
//      case (id1, idf1) =>
//        val idfs: Array[(Long, linalg.Vector)] = b_num_idf_pairs.value.filter(_._1 != id1)
//        val sv1: SV = idf1.asInstanceOf[SV]
//        val bsv1 = new SparseVector[Double](sv1.indices, sv1.values, sv1.size)
//        idfs.map {
//          case (id2, idf2) =>
//            val sv2 = idf2.asInstanceOf[SV]
//            val bsv2 = new SparseVector[Double](sv2.indices, sv2.values, sv2.size)
//            val cosSim = bsv1.dot(bsv2).asInstanceOf[Double] / (norm(bsv1) * norm(bsv2))
//            (id1, id2, cosSim)
//        }

        val docSims = num_idf_pairs.flatMap {
          case (id1, idf1) =>
//            不求自己和自己相似度，所以过滤和自己id一样的
            val idfs: Array[(Long, linalg.Vector)] = b_num_idf_pairs.value.filter(_._1 != id1)
            val sv1: SV = idf1.asInstanceOf[SV]
           val data1 = sv1.toArray
//            遍历所有非当前文档，比较两两之间的相似度
            idfs.map {
              case (id2, idf2) =>
                val sv2 = idf2.asInstanceOf[SV]
                val data2 = sv2.toArray
//                两个向量每个元素相乘之和
                val AB = data1.zip(data2).map(sa=>{sa._1*sa._2}).reduce(_+_)
//                单个向量每个元素平方之和再开根
                val squartA = Math.sqrt(data1.map(sa=>{Math.pow(sa,2)}).reduce(_+_))
                val squartB = Math.sqrt(data2.map(sa=>{Math.pow(sa,2)}).reduce(_+_))
//                带入公式
                val cosSim = AB/(squartA*squartB)
                (id1, id2, cosSim)
            }
    }






    docSims.foreach(println(_))

    sc.stop()

  }
}
