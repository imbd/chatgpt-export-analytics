# 🚀 ChatGPT Export Data Analytics

A comprehensive analysis tool for ChatGPT conversation export data with multi-level visualizations and deep insights.

## 📊 Overview

This project provides a complete analytics pipeline for ChatGPT conversation exports, generating everything from basic statistics to advanced pattern analysis and personality insights.

## ✨ Features

### 📈 Multi-Level Analysis
- **Basic Analytics**: Conversation counts, message statistics, time patterns
- **Model Usage Analysis**: Distribution and evolution of ChatGPT model usage
- **Content Analysis**: Topic extraction, language detection, complexity metrics
- **Advanced Insights**: MBTI personality patterns, decision-making analysis

### 🎨 Rich Visualizations
- Interactive time series charts
- Model usage distribution graphs
- Topic clustering and evolution
- Personality type radar charts
- Decision-making pattern analysis

### 🧠 AI-Powered Insights
- Deep conversation analysis using OpenAI's models
- Semantic topic clustering
- Personal evolution tracking
- Communication pattern recognition

## 🛠️ Technology Stack

- **Python 3.12** - Core language
- **Pandas** - Data processing and analysis
- **Plotly** - Interactive visualizations
- **OpenAI API** - AI-powered analysis
- **Scikit-learn** - Machine learning for clustering
- **UMAP/HDBSCAN** - Advanced clustering algorithms

## 📁 Project Structure

### Core Scripts
- **`structured_analyzer_v3.py`** - Advanced two-phase analysis pipeline
- **`analyze_conversations.py`** - Basic comprehensive analysis tool
- **`quick_insights.py`** - Fast statistics generator

### Analysis Features
- Time series analysis with trends and seasonality
- Model usage patterns and efficiency metrics
- Content analysis with topic extraction
- Communication style and personality profiling

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.8
OpenAI API key
```

### Installation
1. Clone the repository:
```bash
git clone <your-repo-url>
cd chatgpt_export_data
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment:
```bash
cp env_example.txt .env
# Add your OpenAI API key to .env
```

4. Place your ChatGPT export:
```bash
# Place your conversations.json.zip in the project root
```

### Usage

#### Quick Analysis
```bash
python quick_insights.py
```

#### Basic Analysis
```bash
python analyze_conversations.py
```

#### Advanced Analysis
```bash
python structured_analyzer_v3.py
```

## 📊 Output Files

### Basic Analysis
- `dashboard.html` - Main interactive dashboard
- `time_analysis.html` - Temporal patterns and trends
- `model_analysis.html` - Model usage statistics
- `usage_patterns.html` - Behavioral patterns
- `content_analysis.html` - Topic and content analysis

### Advanced Analysis
- `structured_report_v3.html` - Comprehensive report with:
  - MBTI personality analysis
  - Decision-making patterns
  - Life balance wheel
  - Personal evolution tracking

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
```

### Analysis Parameters
- **Sample Size**: Configurable conversation sampling
- **AI Models**: Supports GPT-4, GPT-4o, o1, o3 models
- **Analysis Depth**: Multiple analysis levels available

## 🎯 Use Cases

### For Researchers
- Study human-AI interaction patterns
- Analyze conversation evolution over time
- Investigate linguistic and behavioral trends

### For Product Managers
- Understand user engagement patterns
- Identify popular features and models
- Track usage metrics and trends

### For Personal Insights
- Discover your communication patterns
- Track personal growth and interests
- Analyze decision-making styles

## 📈 Analysis Capabilities

### Temporal Analysis
- Daily, weekly, monthly usage patterns
- Conversation length and complexity trends
- Model adoption patterns over time

### Content Analysis
- Automatic topic extraction and clustering
- Language detection and analysis
- Sentiment and complexity metrics

### Behavioral Analysis
- Communication style profiling
- Decision-making pattern recognition
- Personal evolution tracking

### Advanced Features
- MBTI personality type analysis
- Life balance assessment
- Semantic similarity clustering

## 🔒 Privacy & Security

- All analysis runs locally on your machine
- No data is sent to external services (except OpenAI for analysis)
- Personal data remains in your control
- Comprehensive .gitignore prevents accidental data exposure

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source. Please ensure you comply with OpenAI's API usage terms when using the AI analysis features.

## ⚠️ Important Notes

- Requires valid OpenAI API key for advanced analysis
- Large conversation datasets may take time to process
- Ensure you have sufficient API credits for comprehensive analysis
- The tool respects OpenAI API rate limits

## 🆘 Troubleshooting

### Common Issues
- **API Key Error**: Ensure your OpenAI API key is correctly set in `.env`
- **Rate Limits**: The tool includes automatic rate limiting
- **Memory Issues**: Large datasets may require sampling (automatically handled)

### Performance Tips
- Use quick_insights.py for fast overview
- Configure sample sizes for large datasets
- Monitor API usage during analysis

---

*Built with Python, data science, and AI-powered insights* ❤️