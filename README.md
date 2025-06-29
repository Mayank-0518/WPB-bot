# WhatsApp AI Second Brain Assistant

A sophisticated AI-powered personal assistant that combines WhatsApp integration with document analysis, task management, and intelligent question-answering using IBM Granite-3-3-8b and vector storage.

## 🚀 Features

- **WhatsApp Integration**: Chat with your AI assistant via WhatsApp using Twilio
- **Document Processing**: Upload and analyze PDFs, text files, and other documents  
- **Vector Search**: FAISS-powered semantic search across your knowledge base
- **Task & Reminder Extraction**: Automatically extract tasks and reminders from conversations
- **Scheduled Notifications**: Send WhatsApp reminders at specified times
- **RAG Q&A**: Answer questions based on your personal knowledge base
- **Text Summarization**: Summarize long documents and conversations
- **MongoDB Storage**: Persistent storage for conversations, tasks, and metadata

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.9+
- **AI**: IBM Granite-3-3-8b via watsonx.ai (with IAM authentication)
- **Vector Database**: FAISS for semantic search
- **WhatsApp**: Twilio WhatsApp API
- **Database**: MongoDB for persistent storage
- **Scheduling**: APScheduler for reminders
- **Document Processing**: PyMuPDF, python-docx



## 📋 Prerequisites

1. **IBM watsonx.ai Account**:
   - Create an IBM Cloud account
   - Get API key from IBM Cloud IAM
   - Create a watsonx.ai project and note the Project ID
   - Ensure you have access to Granite-3-3-8b-instruct model

2. **Twilio Account**:
   - Sign up for Twilio
   - Set up WhatsApp Sandbox
   - Get Account SID, Auth Token, and Phone Number



## ⚙️ Setup Instructions

### 1. Clone and Install

```bash
git clone <repository-url>
cd "WP brain"
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

### 3. IBM watsonx.ai Authentication Setup

The system uses **IAM token authentication** for secure watsonx.ai access:

1. **Get IBM Cloud API Key**:
   - Go to IBM Cloud Console → Manage → Access (IAM) → API keys
   - Create a new API key and copy it

2. **Get Project ID**:
   - Open your watsonx.ai project
   - Project ID is visible in the project URL or settings

3. **Configure Environment**:
```bash
# IBM Granite (watsonx.ai)
GRANITE_API_KEY=your_ibm_cloud_api_key_here
GRANITE_PROJECT_ID=your_watsonx_project_id_here
GRANITE_API_URL=https://us-south.ml.cloud.ibm.com
IBM_MODEL_ID=granite-3-3-8b-instruct
```

**Authentication Flow**:
- System automatically exchanges API key for IAM tokens
- Tokens are cached and refreshed automatically before expiration
- All watsonx.ai requests use Bearer token authentication
- Failed auth attempts trigger automatic token refresh

### 4. Twilio WhatsApp Setup

```bash
# Twilio (WhatsApp Sandbox)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=whatsapp:+1415xxxxxxx
```



### 5. Run the Application

```bash
# Start the FastAPI server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# The API will be available at:
# http://localhost:8000
# Documentation: http://localhost:8000/docs
```

## 📱 WhatsApp Usage

### Connect to WhatsApp Sandbox

1. **Join Sandbox**:
   - Go to Twilio Console → WhatsApp → Sandbox
   - Send the join code to the sandbox number from your WhatsApp

2. **Configure Webhook**:
   - Set webhook URL: `https://your-domain.com/whatsapp/webhook`
   - Use ngrok for local development: `ngrok http 8000`

### Message Types

**📝 Save Notes/Documents**:
```
"Remember this: Machine learning is a subset of AI..."
"Note: Meeting with John tomorrow at 3pm"
```

**❓ Ask Questions**:
```
"What did I save about machine learning?"
"Tell me about my meeting with John"
```

**📋 Task Management**:
```
"I need to call the client tomorrow and submit the report by Friday"
"Remind me to pay bills next week"
```

**📄 Document Summarization**:
```
"Summarize this: [long text or document]"
```

## 🔧 API Endpoints

### Core Endpoints

- `POST /whatsapp/webhook` - WhatsApp message webhook
- `POST /whatsapp/send` - Send WhatsApp message
- `POST /documents/upload` - Upload document
- `POST /ai/summarize` - Summarize text
- `POST /ai/extract-tasks` - Extract tasks from text
- `POST /ai/answer` - RAG-based Q&A
- `GET /reminders/` - List reminders
- `POST /reminders/` - Create reminder

### Authentication Testing

Test IAM token authentication:

```bash
# Test token generation
python -c "
import asyncio
from backend.utils.iam_token import test_iam_token
asyncio.run(test_iam_token())
"

# Test Granite API
python backend/ai/granite_api.py
```

## 🧪 Testing & Development

### Production Tests
Located in `test/` directory - these are included in the repository:

```bash
# Run all production tests
python -m pytest test/ -v

# Test specific components  
python test/test_ai.py
```



### Test WhatsApp Integration

1. **Send Test Message**:
```bash
curl -X POST "http://localhost:8000/whatsapp/send" \
  -H "Content-Type: application/json" \
  -d '{
    "to": "whatsapp:+1234567890",
    "message": "Hello from AI Assistant!"
  }'
```

2. **Test Document Upload**:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@example_docs/sample.pdf"
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GRANITE_API_KEY` | IBM Cloud API Key | Required |
| `GRANITE_PROJECT_ID` | watsonx.ai Project ID | Required |
| `GRANITE_API_URL` | watsonx.ai API URL | us-south endpoint |
| `IBM_MODEL_ID` | Model identifier | granite-3-3-8b-instruct |
| `TWILIO_ACCOUNT_SID` | Twilio Account SID | Required |
| `TWILIO_AUTH_TOKEN` | Twilio Auth Token | Required |
| `TWILIO_PHONE_NUMBER` | WhatsApp Phone Number | Required |
| `MONGO_URI` | MongoDB connection string | Required |
| `DEBUG` | Enable debug mode | True |
| `MAX_FILE_SIZE` | Max upload size (bytes) | 10MB |

### Model Configuration

The system uses IBM Granite-3-3-8b-instruct with optimized parameters:

- **Temperature**: 0.7 (balanced creativity)
- **Max Tokens**: 500 (responses)
- **Top-p**: 0.9 (nucleus sampling)
- **Stop Sequences**: `["</s>", "\n\n---"]`

## 🔍 Troubleshooting

### Common Issues

1. **IAM Authentication Errors**:
   ```
   Authentication failed - check IAM token and project ID
   ```
   - Verify `GRANITE_API_KEY` is a valid IBM Cloud API key
   - Ensure `GRANITE_PROJECT_ID` matches your watsonx.ai project
   - Check API key has watsonx.ai access permissions

2. **Token Refresh Issues**:
   ```
   Failed to fetch IAM token
   ```
   - Check internet connectivity
   - Verify IBM Cloud API key is active
   - Check API key hasn't expired

3. **WhatsApp Webhook Issues**:
   ```
   Webhook verification failed
   ```
   - Ensure webhook URL is accessible
   - Check Twilio webhook configuration
   - Verify HTTPS for production

4. **Document Processing Errors**:
   ```
   File upload failed
   ```
   - Check file size limits
   - Ensure supported file format
   - Verify upload directory permissions











