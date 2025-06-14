# Customer Support AI Assistant

## Project Overview

This is a complete AI project that fine-tunes TinyLlama 1.1B for customer support using LoRA/QLoRA techniques. The project includes a web interface built with Flask and comprehensive analytics dashboard.

## 🚀 Key Features

- **Fine-tuned TinyLlama 1.1B**: Efficient 1.1 billion parameter model optimized for customer support
- **Parameter-Efficient Training**: Uses LoRA/QLoRA for memory-efficient fine-tuning
- **Web Interface**: Professional Flask application with real-time chat
- **Analytics Dashboard**: Performance metrics and conversation analysis
- **Real-time Inference**: Fast response generation with detailed metrics
- **Visualization**: Training curves and performance comparisons

## 🛠 Technical Stack

### Core AI Components
- **Model**: TinyLlama 1.1B (unsloth/tinyllama-chat-bnb-4bit)
- **Fine-tuning**: Hugging Face Transformers + LoRA (via PEFT)
- **Training Framework**: Transformers + TRL


### Web Application
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Plotly.js for interactive charts
- **UI Framework**: Modern responsive design

### Data Processing
- **Dataset**: Customer support conversations (Bitext + custom samples)
- **Format**: JSONL instruction-following format
- **Categories**: Account, Billing, Technical Support, Shipping, Returns, Product Info

## 📁 Project Structure

```
customer_support_ai/
├── app.py                 # Flask web application
├── fine_tune_model.py     # Model fine-tuning script
├── inference.py           # Inference engine
├── data_processor.py      # Data processing utilities
├── config.py             # Configuration settings
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # Frontend JavaScript
├── templates/
│   ├── index.html        # Main chat interface
│   ├── analytics.html    # Analytics dashboard
│   └── results.html      # Results page
├── data/
│   ├── sample_data.jsonl # Sample training data
│   └── README.md         # Data documentation
└── models/
    ├── tinyllama-customer-support/ # Fine-tuned model
    └── README.md         # Model documentation
```

## 🔧 Installation & Setup

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd customer_support_ai

# Install dependencies
pip install -r requirements.txt

# For GPU training (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Data

```bash
# The data processor will automatically create sample data
python data_processor.py
```

### 3. Fine-tune the Model

```bash
# Start fine-tuning (takes ~30-60 minutes on GPU)
python fine_tune_model.py
```

### 4. Run the Web Application

```bash
# Start Flask server
python app.py
```

Visit `http://localhost:5000` to access the web interface.

## 🎯 Real-World Problem Solved

### Problem: Customer Support Automation
- **Challenge**: Manual customer support is expensive and slow
- **Solution**: AI assistant that provides instant, accurate responses
- **Impact**: 24/7 availability, consistent quality, reduced response time

### Use Cases
1. **Account Management**: Login issues, password resets, profile updates
2. **Billing Support**: Payment problems, refunds, subscription questions
3. **Technical Support**: App troubleshooting, connectivity issues
4. **Order Management**: Shipping tracking, delivery issues, returns

## 📊 Model Performance

### Training Metrics
- **Base Model**: TinyLlama 1.1B parameters
- **Trainable Parameters**: ~16M (1.16% of total)
- **Training Time**: ~30 minutes on T4 GPU


## 🔬 Technical Implementation

### Fine-tuning Configuration
```python
# LoRA Configuration
LORA_CONFIG = {
    "r": 16,                    # Rank
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
}

# Training Configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 9000, # almost 3 epoch
    "learning_rate": 2e-4,
    "fp16": True,
    .....
}
```

### Data Format
```json
{
    "instruction": "Customer inquiry about Account - Login Issues: I can't access my account",
    "response": "I understand how frustrating this can be. Let me help you resolve this login issue...",
    "category": "Account",
    "input": ""
}
```

## 📈 Data Sources & Acquisition

### Primary Dataset
- **Source**: Bitext Customer Support Dataset
- **URL**: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **Size**: 27,000+ professional customer support examples
- **License**: Commercial use allowed

### Sample Data Creation
The project includes a data processor that creates realistic sample data:
```python
from data_processor import CustomerSupportDataProcessor
processor = CustomerSupportDataProcessor()
processor.create_sample_data()  # Creates 8 diverse examples
processor.download_real_dataset()  # Downloads Bitext dataset
```

### Data Categories
1. **Account Issues** (35%): Login problems, profile management
2. **Technical Support** (25%): App issues, connectivity problems
3. **Billing** (18%): Payment issues, refunds, subscriptions
4. **Shipping** (12%): Order tracking, delivery problems
5. **Returns** (7%): Return processes, refund requests
6. **Product Information** (3%): Feature questions, specifications

## 🚀 Running the Complete System

### Quick Start (Demo Mode)
```bash
# Run without fine-tuning (uses demo responses)
python app.py
```

### Full Training Pipeline
```bash
# 1. Process data
python -c "from data_processor import CustomerSupportDataProcessor; CustomerSupportDataProcessor().create_sample_data()"

# 2. Fine-tune model
python fine_tune_model.py

# 3. Test inference
python inference.py demo

# 4. Start web application
python app.py
```

### Web Interface Features
- **Chat Interface**: Real-time conversation with AI
- **Category Selection**: Specify inquiry type for better responses
- **Analytics Dashboard**: Performance metrics and visualizations
- **Response Metrics**: Track generation time and quality
- **Conversation History**: Review past interactions

## 📊 Visualization & Analytics

The web application includes comprehensive analytics:

### Training Visualizations
- Loss curves during fine-tuning
- Model convergence metrics
- Training progress tracking

### Performance Metrics
- Response time distribution
- Category-wise inquiry analysis
- Conversation volume over time
- Customer satisfaction scoring

### Real-time Monitoring using [wanbd.ai](https://wandb.ai/home)
- Live performance statistics
- Memory usage tracking
- Model health monitoring

## 🎨 Web Interface Design

### Modern UI Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Chat**: WebSocket-like experience with AJAX
- **Interactive Charts**: Plotly.js visualizations
- **Professional Styling**: Bootstrap 5 + custom CSS
- **Loading Indicators**: Smooth user experience
- **Error Handling**: Graceful failure modes

### Accessibility
- WCAG 2.1 AA compliant design
- Keyboard navigation support
- Screen reader compatibility
- High contrast mode support

## 🔮 Extension Possibilities

### Model Improvements
- Multi-language support
- Larger context windows
- Integration with RAG systems
- Voice input/output capabilities

### Application Features
- User authentication
- Conversation persistence
- Integration with CRM systems
- A/B testing framework
- Advanced analytics

### Deployment Options
- Docker containerization
- Cloud deployment (AWS, GCP, Azure)
- Edge deployment
- API service conversion

## 📝 License & Usage

This project is designed for educational and commercial use. Key components:

- **TinyLlama**: Apache 2.0 License
- **Training Data**: Bitext dataset (commercial use allowed)
- **Code**: MIT License (free for commercial use)
- **Web Interface**: MIT License

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Model Performance**: Better training strategies
2. **UI/UX**: Enhanced user interface
3. **Data Quality**: More diverse training examples
4. **Documentation**: Additional tutorials and guides
5. **Testing**: Comprehensive test suite

## 📚 References & Resources

### Documentation
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [TinyLlama Paper](https://arxiv.org/abs/2401.02385)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

### Datasets
- [Bitext Customer Support Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Customer Support Conversations](https://www.kaggle.com/datasets/bitext/training-dataset-for-chatbotsvirtual-assistants)

### Tools & Frameworks
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Unsloth Training Framework](https://github.com/unslothai/unsloth)
- [Flask Web Framework](https://flask.palletsprojects.com/)
- [Plotly Visualization](https://plotly.com/python/)

---

## 🎉 Project Highlights

- ✅ **Complete End-to-End Pipeline**: From data processing to deployment
- ✅ **Production-Ready**: Professional web interface and monitoring
- ✅ **Efficient Training**: LoRA/qLoRA for fast, memory-efficient fine-tuning
- ✅ **Real-World Application**: Solves actual customer support challenges
- ✅ **Comprehensive Documentation**: Detailed setup and usage instructions
- ✅ **Modern Tech Stack**: Latest AI and web development frameworks
- ✅ **Scalable Architecture**: Easy to extend and customize


This project demonstrates how to build a complete AI application using modern techniques, solving real business problems with efficient small language models.
