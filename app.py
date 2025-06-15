from flask import Flask, render_template, request, jsonify, session
import json
import time
import plotly
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import os
from inferance import CustomerSupportInference  # Fixed typo: inferance -> inference
from config import FLASK_CONFIG

app = Flask(__name__)
app.secret_key = FLASK_CONFIG["SECRET_KEY"]

# Initialize the inference engine
inference_engine = None


def get_inference_engine():
    """Lazy load the inference engine"""
    global inference_engine
    if inference_engine is None:
        inference_engine = CustomerSupportInference()
        try:
            inference_engine.load_model()
        except Exception as e:
            print(f"Warning: Could not load fine-tuned model: {e}")
            print("The application will run in demo mode.")
    return inference_engine


# Sample data for demonstration
sample_conversations = []
response_times = []
categories_count = {}


@app.route('/')
def index():
    """Main page with chat interface"""
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages and generate AI responses"""
    try:
        data = request.get_json()
        customer_query = data.get('message', '').strip()
        category = data.get('category', '').strip() or None

        if not customer_query:
            return jsonify({'error': 'Please enter a message'}), 400

        # Get inference engine
        engine = get_inference_engine()

        # Generate response
        start_time = time.time()

        if engine.model is None:
            # Demo mode - return sample response
            response_text = generate_demo_response(customer_query, category)
            generation_time = 0.5
            result = {
                'response': response_text,
                'generation_time': generation_time,
                'prompt_length': len(customer_query),
                'response_length': len(response_text.split())
            }
        else:
            # Use actual model
            result = engine.generate_response(customer_query, category)

        # Store conversation for analytics
        conversation = {
            'timestamp': datetime.now().isoformat(),
            'customer_query': customer_query,
            'category': category or 'General',
            'ai_response': result['response'],
            'generation_time': result['generation_time'],
            'response_length': result.get('response_length', 0)
        }

        sample_conversations.append(conversation)
        response_times.append(result['generation_time'])

        # Update category counts
        cat = category or 'General'
        categories_count[cat] = categories_count.get(cat, 0) + 1

        # Keep only last 100 conversations
        if len(sample_conversations) > 100:
            sample_conversations.pop(0)
            response_times.pop(0)

        return jsonify({
            'response': result['response'],
            'generation_time': result['generation_time'],
            'category': category or 'General',
            'conversation_id': len(sample_conversations)
        })

    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500


def generate_demo_response(query, category):
    """Generate demo responses when model is not available"""
    demo_responses = {
        'Account': "Thank you for contacting us about your account issue. I'd be happy to help you resolve this. Let me check your account details and guide you through the solution.",
        'Billing': "I understand your billing concern and I'm here to help resolve this for you. Let me review your account and transaction history to provide you with the best solution.",
        'Technical Support': "I'm sorry to hear you're experiencing technical difficulties. Let me help you troubleshoot this issue step by step to get everything working properly.",
        'Shipping': "I understand your concern about shipping. Let me track your order and provide you with an update on the delivery status.",
        'Returns': "I'm happy to assist you with your return request. Let me walk you through our return process and help make this as smooth as possible.",
        'General': "Thank you for reaching out to our customer support. I'm here to help you with any questions or concerns you may have. How can I assist you today?"
    }

    base_response = demo_responses.get(category, demo_responses['General'])

    # Add some query-specific context
    if 'password' in query.lower():
        base_response += " For password issues, I recommend using our secure password reset link."
    elif 'refund' in query.lower():
        base_response += " I'll be happy to process your refund request once I verify the details."
    elif 'cancel' in query.lower():
        base_response += " I can help you with the cancellation process and explain any applicable policies."

    return base_response


@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')


@app.route('/api/analytics')
def get_analytics():
    """API endpoint for analytics data"""
    try:
        # Response time chart
        response_time_fig = create_response_time_chart()

        # Category distribution chart
        category_fig = create_category_chart()

        # Conversation volume over time
        volume_fig = create_volume_chart()

        # Calculate summary statistics
        stats = calculate_stats()

        return jsonify({
            'response_time_chart': response_time_fig,
            'category_chart': category_fig,
            'volume_chart': volume_fig,
            'stats': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def create_response_time_chart():
    """Create response time distribution chart"""
    if not response_times:
        return {}

    fig = px.histogram(
        x=response_times,
        nbins=20,
        title="AI Response Time Distribution",
        labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
    )
    fig.update_layout(
        height=400,
        template="plotly_white"
    )

    return json.loads(fig.to_json())


def create_category_chart():
    """Create category distribution pie chart"""
    if not categories_count:
        return {}

    fig = px.pie(
        values=list(categories_count.values()),
        names=list(categories_count.keys()),
        title="Customer Inquiries by Category"
    )
    fig.update_layout(height=400)

    return json.loads(fig.to_json())


def create_volume_chart():
    """Create conversation volume over time chart"""
    if not sample_conversations:
        return {}

    # Create hourly conversation counts
    df = pd.DataFrame(sample_conversations)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.floor('H')

    hourly_counts = df.groupby('hour').size().reset_index(name='count')

    fig = px.line(
        hourly_counts,
        x='hour',
        y='count',
        title="Conversation Volume Over Time",
        labels={'hour': 'Time', 'count': 'Number of Conversations'}
    )
    fig.update_layout(height=400)

    return json.loads(fig.to_json())


def calculate_stats():
    """Calculate summary statistics"""
    if not sample_conversations:
        return {
            'total_conversations': 0,
            'avg_response_time': 0,
            'avg_response_length': 0,
            'most_common_category': 'N/A'
        }

    total_conversations = len(sample_conversations)
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    response_lengths = [conv.get('response_length', 0) for conv in sample_conversations]
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    most_common_category = max(categories_count.items(), key=lambda x: x[1])[0] if categories_count else 'N/A'

    return {
        'total_conversations': total_conversations,
        'avg_response_time': round(avg_response_time, 2),
        'avg_response_length': round(avg_response_length, 1),
        'most_common_category': most_common_category
    }


@app.route('/api/conversations')
def get_conversations():
    """Get recent conversations"""
    # Return last 10 conversations
    recent_conversations = sample_conversations[-10:] if sample_conversations else []
    return jsonify(recent_conversations)


@app.route('/test')
def test_model():
    """Test endpoint for model evaluation"""
    try:
        engine = get_inference_engine()

        if engine.model is None:
            return jsonify({
                'status': 'demo_mode',
                'message': 'Running in demo mode - fine-tuned model not available'
            })

        # Run model evaluation
        results = engine.evaluate_model()

        return jsonify({
            'status': 'success',
            'evaluation_results': results
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("Starting Customer Support AI Web Application...")
    print(f"Access the app at: http://localhost:{FLASK_CONFIG['PORT']}")

    app.run(
        host=FLASK_CONFIG["HOST"],
        port=FLASK_CONFIG["PORT"],
        debug=FLASK_CONFIG["DEBUG"]
    )