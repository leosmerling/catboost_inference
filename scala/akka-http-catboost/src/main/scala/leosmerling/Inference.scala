package leosmerling

//#user-registry-actor
import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import scala.collection.immutable

import ai.catboost.CatBoostModel
import ai.catboost.CatBoostPredictions

case class PredictionResult(
    predicted_days: Float
)

case class InputFeatures(
    f1: Float,
    f2: Float,
    catf1: String, 
    catf2: String, 
    catf3: String, 
)

object Inference {
    //actor protocol
    sealed trait Command

    final case class Predict(features: InputFeatures, replyTo: ActorRef[GetPrediction]) extends Command

    final case class GetPrediction(result: PredictionResult)

    def loadModel(): CatBoostModel = {
        println("Loading model...")
        CatBoostModel.loadModel("src/main/resources/model.cbm")
    }

    def apply() = predict(loadModel())

    def predict(model: CatBoostModel): Behavior[Command] = 
        Behaviors.receiveMessage {
            case Predict(features, replyTo) =>
                val predictions = model.predict(
                    Array[Float](features.f1, features.f2),
                    Array[String](features.catf1, features.catf2, features.catf3)
                )
                println(s"Predictions: {predictions}")
                replyTo ! GetPrediction(result = PredictionResult(predictions.get(0,0).toFloat))
                Behaviors.same
        }
}
