/////////////////////////////////////////////////////////////////////////////////////////////////
//NeuralNetwork Class
/////////////////////////////////////////////////////////////////////////////////////////////////

//the class to define the initialisation and operations of the overall NN model
public class NeuralNetwork {

	NeuralNetworkLayer	InputLayer;
	NeuralNetworkLayer	HiddenLayer;
	NeuralNetworkLayer	OutputLayer;

//constructor - to initialise each layer
public NeuralNetwork(){
	
	InputLayer = new NeuralNetworkLayer();
	HiddenLayer = new NeuralNetworkLayer();
	OutputLayer = new NeuralNetworkLayer();
	
	}
//the initialisation of the structure of the neural network
public void Initialize(int nNodesInput, int nNodesHidden, int nNodesOutput) //the number of nodes for each layer (input, hidden and output)
{
	//the initialise of the input layer
    InputLayer.NumberOfNodes = nNodesInput;
    InputLayer.NumberOfChildNodes = nNodesHidden;
    InputLayer.NumberOfParentNodes = 0;
    InputLayer.Initialize(nNodesInput, null, HiddenLayer);
    InputLayer.RandomizeWeights();

    //the initialise of the hidden layer
    HiddenLayer.NumberOfNodes = nNodesHidden;
    HiddenLayer.NumberOfChildNodes = nNodesOutput;
    HiddenLayer.NumberOfParentNodes = nNodesInput;
    HiddenLayer.Initialize(nNodesHidden, InputLayer, OutputLayer);
    HiddenLayer.RandomizeWeights();

    //the initialise of the output layer
    OutputLayer.NumberOfNodes = nNodesOutput;
    OutputLayer.NumberOfChildNodes = 0;
    OutputLayer.NumberOfParentNodes = nNodesHidden;
    OutputLayer.Initialize(nNodesOutput, HiddenLayer, null);

}

//set the input of the neural network for each instance
public void	SetInput(int i, double value)
{
    if((i>=0) && (i<InputLayer.NumberOfNodes))
    {
       InputLayer.NeuronValues[i] = value;
    }
}
//get the output of the NN for each instance
public double GetOutput(int i)
{
    if((i>=0) && (i<OutputLayer.NumberOfNodes))
    {
      return OutputLayer.NeuronValues[i];
    }

    return (double) 10000; // to indicate an error
}

//set the desired output for each node in the output layer 
public void SetDesiredOutput(int i, double value)
{
    if((i>=0) && (i<OutputLayer.NumberOfNodes))
    {
      OutputLayer.DesiredValues[i] = value;
    }
}
//get the real output for each instance
public void FeedForward()
{
    InputLayer.CalculateNeuronValues();
    HiddenLayer.CalculateNeuronValues();
    OutputLayer.CalculateNeuronValues();
}
//identify the error and adjust the weight
public void BackPropagate()
{
    OutputLayer.CalculateErrors();
    HiddenLayer.CalculateErrors();

    HiddenLayer.AdjustWeights();
    InputLayer.AdjustWeights();
}
//get the class with the highest probability as the final prediction result
public int	GetMaxOutputID()
{
    int		i, id;
    double	maxval;

    maxval = OutputLayer.NeuronValues[0];
    id = 0;

    for(i=1; i<OutputLayer.NumberOfNodes; i++)
    {
         if(OutputLayer.NeuronValues[i] > maxval)
          {
             maxval = OutputLayer.NeuronValues[i];
             id = i;
          }
     }

     return id;
}
//identify the error for each node in the layer
public double CalculateError()
{
     int		i;
     double	error = 0;

     for(i=0; i<OutputLayer.NumberOfNodes; i++)
     {
         error += Math.pow(OutputLayer.NeuronValues[i] - OutputLayer.DesiredValues[i], 2);
     }

     error = error / OutputLayer.NumberOfNodes;

     return error;
}
//set the learning rate
public void SetLearningRate(double rate)
{
     InputLayer.LearningRate = rate;
     HiddenLayer.LearningRate = rate;
     OutputLayer.LearningRate = rate;
}
//set the linear output for each instance - not used in this exercise
public void	SetLinearOutput(boolean useLinear)
{
     InputLayer.LinearOutput = useLinear;
     HiddenLayer.LinearOutput = useLinear;
     OutputLayer.LinearOutput = useLinear;
}
//set other learning parameters
public void	SetMomentum(boolean useMomentum, double factor)
{
     InputLayer.UseMomentum = useMomentum;
     HiddenLayer.UseMomentum = useMomentum;
     OutputLayer.UseMomentum = useMomentum;

     InputLayer.MomentumFactor = factor;
     HiddenLayer.MomentumFactor = factor;
     OutputLayer.MomentumFactor = factor;

}
//save the weights in a file
public void DumpData()
{

     int		i, j;

     System.out.println("--------------------------------------------------------");
     System.out.println( "Input Layer");
     System.out.println("--------------------------------------------------------");
     System.out.println("\n");
     System.out.println( "Node Values:");
     System.out.println("\n");
     
     for(i=0; i<InputLayer.NumberOfNodes; i++)
          System.out.println( i + " " + InputLayer.NeuronValues[i]);
     System.out.println( "\n");
     System.out.println("Weights:");
     System.out.println("\n");
     
     for(i=0; i<InputLayer.NumberOfNodes; i++)
          for(j=0; j<InputLayer.NumberOfChildNodes; j++)
             System.out.println(i + " " + j + " " + InputLayer.Weights[i][j]);
     System.out.println("\n");
     System.out.println("Bias Weights:");
     System.out.println("\n");
     
     for(j=0; j<InputLayer.NumberOfChildNodes; j++)
          System.out.println( j + " " + InputLayer.BiasWeights[j]);

     System.out.println( "\n");
     System.out.println("\n");

     System.out.println( "--------------------------------------------------------");
     System.out.println("Hidden Layer");
     System.out.println( "--------------------------------------------------------");
     System.out.println("\n");
     System.out.println( "Weights:");
     System.out.println( "\n");
     
     for(i=0; i<HiddenLayer.NumberOfNodes; i++)
         for(j=0; j<HiddenLayer.NumberOfChildNodes; j++)
              System.out.println( i + " " + j + " " + HiddenLayer.Weights[i][j]);
     System.out.println( "\n");
     System.out.println("Bias Weights:");
     System.out.println( "\n");
     
     for(j=0; j<HiddenLayer.NumberOfChildNodes; j++)
         System.out.println( j + " " + HiddenLayer.BiasWeights[j]);

     System.out.println( "\n");
     System.out.println( "\n");

     System.out.println( "--------------------------------------------------------");
     System.out.println( "Output Layer");
     System.out.println( "--------------------------------------------------------");
     System.out.println( "\n");
     System.out.println( "Node Values:");
     System.out.println( "\n");
     
     for(i=0; i<OutputLayer.NumberOfNodes; i++)
         System.out.println(i + " " + OutputLayer.NeuronValues[i]);
     System.out.println( "\n");

  }


}
