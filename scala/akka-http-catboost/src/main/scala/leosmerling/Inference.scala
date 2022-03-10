package leosmerling

//#user-registry-actor
import akka.actor.typed.ActorRef
import akka.actor.typed.Behavior
import akka.actor.typed.scaladsl.Behaviors
import scala.collection.immutable

case class PredictionResult(
    predicted_days: Float
)

case class InputFeatures(f1: String, f2: Float)

object Inference {
    //actor protocol
    sealed trait Command
    final case class Predict(features: InputFeatures, replyTo: ActorRef[GetPrediction]) extends Command

    final case class GetPrediction(result: PredictionResult)

    def apply(): Behavior[Command] = 
        Behaviors.receiveMessage {
            case Predict(features, replyTo) => 
                replyTo ! GetPrediction(result = PredictionResult(features.f2))
                Behaviors.same
        }
}
