# Domain Expert LLM: Financial Analysis Specialist

## Project Overview
This project demonstrates the fine-tuning of Meta's Llama 2 7B foundation model to create a domain-specific expert model in financial analysis using AWS SageMaker. The fine-tuned model serves as a specialized knowledge base capable of generating accurate, contextually relevant responses for financial analysis tasks.

## 🎯 Objective
The primary goal of this project was to develop a proof of concept (POC) for a financial domain expert model that can be utilized in various applications:
- Interactive financial advisory chat applications
- Internal financial knowledge management systems 
- Automated financial content generation for company materials
- Investment analysis support tools

## 🔍 Solution Architecture
![Architecture Diagram](architecture_diagram.png)

The solution leverages AWS SageMaker to:
1. Deploy the base Meta Llama 2 7B foundation model
2. Fine-tune the model on domain-specific data
3. Deploy and test the resulting domain expert model

## 🛠️ Technical Implementation

### AWS Services Used
- **Amazon SageMaker**: For model deployment, fine-tuning, and hosting
- **Amazon S3**: For storing datasets and model artifacts
- **AWS IAM**: For managing access permissions
- **CloudWatch**: For monitoring model performance and costs

### Model Details
- **Base Model**: Meta Llama 2 7B
- **Fine-tuning Method**: Parameter-efficient fine-tuning (PEFT) with LoRA
- **Domain Dataset**: Financial domain dataset from AWS project bucket
- **Training Parameters**:
  - Learning rate: 0.0001
  - Batch size: 4
  - Training epochs: 5
  - LoRA rank (r): 8
  - LoRA alpha: 32
  - LoRA dropout: 0.05
  - Target modules: q_proj, v_proj

## 📊 Results & Evaluation

### Performance Comparison
The model evaluation showed significant improvement in domain-specific knowledge after fine-tuning:

| Model Stage | Response Quality | Domain Expertise |
|-------------|------------------|------------------|
| Base Model | Generic, theoretical | Basic financial terminology |
| Fine-tuned Model | Specific, practical | Advanced financial analysis concepts |

### Sample Outputs

**Input prompt**: "The investment tests performed indicate"

**Before fine-tuning**:
```
that the proposed methodology is able to detect and avoid investment frauds, and to select investments with a high potential of profit. The proposed methodology is based on a multi-criteria decision-making approach. The decision-making process is performed by means of the Analytic Hierarchy Process
```

**After fine-tuning**:
```
that the use of the proposed methodology is a very effective tool for the analysis of the investment portfolio. The methodology is based on the use of a set of indicators that, together, provide a comprehensive view of the portfolio. The indicators are classified into four categories: 1)
```

The fine-tuned model demonstrates more structured, detailed financial knowledge with industry-specific analysis terminology.

## 💡 Key Learnings
- Parameter-efficient fine-tuning (PEFT) using LoRA provides an effective approach for specializing large language models with limited computational resources
- Financial domain adaptation requires careful selection of training parameters to balance general language capabilities with specialized knowledge
- Domain-specific evaluation prompts are crucial for measuring improvements in model expertise
- AWS SageMaker provides a robust platform for managing the entire LLM fine-tuning lifecycle
- Proper cost management and resource cleanup are essential when working with large models on AWS

## 🚀 Getting Started

### Prerequisites
- AWS account with appropriate permissions
- Python 3.8+
- Familiarity with AWS SageMaker and Jupyter notebooks

### Installation & Setup
1. Clone this repository:
```bash
git clone https://github.com/yourusername/financial-domain-expert-llm.git
cd financial-domain-expert-llm
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials:
```bash
aws configure
```

4. Set up a SageMaker notebook instance via AWS console:
   - Instance type: ml.t3.medium (for setup) or ml.g5.2xlarge (for training)
   - IAM role with appropriate SageMaker permissions
   - Attach the project Git repository (optional)

### Running the Project
1. Open the Jupyter notebook `Model_Evaluation.ipynb` to deploy and evaluate the base model
2. Proceed to `Model_FineTuning.ipynb` to execute the fine-tuning process and test the fine-tuned model
3. Follow all instructions within notebooks for proper resource cleanup

## 📁 Project Structure
```
financial-domain-expert-llm/
├── notebooks/
│   ├── Model_Evaluation.ipynb        # Base model deployment and evaluation
│   └── Model_FineTuning.ipynb        # Fine-tuning and final model evaluation
├── docs/
│   └── UDACITY_Project_Documentation_Report.docx  # Project documentation
├── images/
│   └── architecture_diagram.png      # AWS architecture visualization
├── requirements.txt                  # Project dependencies
└── README.md                         # Project overview and instructions
```

## ⚠️ Cost Management
This project was developed with a budget constraint of $25 for AWS resources. Cost management strategies included:
- Shutting down resources when not in use (critical after each notebook section)
- Optimizing instance types based on workload requirements (ml.g5.2xlarge for training)
- Using parameter-efficient fine-tuning (LoRA) to reduce computational requirements
- Cleaning up deployments and endpoints immediately after testing
- Following structured cleanup procedures in notebook cells
- Monitoring resource usage through AWS CloudWatch

## 🔮 Future Work
- Explore multi-task fine-tuning across related financial subdomains (investment analysis, risk assessment, market forecasting)
- Implement retrieval-augmented generation (RAG) for improved factual accuracy with company-specific financial data
- Develop a user-friendly web interface for financial analysts to interact with the model
- Benchmark against commercial financial analysis solutions
- Expand training data with more diverse financial documents and market reports
- Implement additional evaluation metrics specific to financial analysis quality

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Meta for providing access to the Llama 2 model family
- AWS for the SageMaker platform and educational credits
- Udacity for the project framework and guidance
- The financial domain dataset providers
