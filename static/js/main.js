// Customer Support AI Frontend JavaScript

class CustomerSupportApp {
    constructor() {
        this.conversationCount = 0;
        this.responseTimes = [];
        this.isLoading = false;

        this.initializeElements();
        this.attachEventListeners();
        this.checkModelStatus();
    }

    initializeElements() {
        this.chatMessages = document.getElementById('chat-messages');
        this.messageInput = document.getElementById('message-input');
        this.categorySelect = document.getElementById('category-select');
        this.sendButton = document.getElementById('send-button');
        this.chatForm = document.getElementById('chat-form');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));

        // Performance stats elements
        this.conversationCountEl = document.getElementById('conversation-count');
        this.avgResponseTimeEl = document.getElementById('avg-response-time');
        this.modelStatusEl = document.getElementById('model-status');
    }

    attachEventListeners() {
        // Chat form submission
        this.chatForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });

        // Sample question buttons
        document.querySelectorAll('.sample-question').forEach(button => {
            button.addEventListener('click', (e) => {
                const question = e.target.dataset.question;
                const category = e.target.dataset.category;

                this.messageInput.value = question;
                this.categorySelect.value = category;
                this.sendMessage();
            });
        });

        // Enter key handling
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea (if needed)
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = this.messageInput.scrollHeight + 'px';
        });
    }

    async checkModelStatus() {
        try {
            const response = await fetch('/test');
            const data = await response.json();

            if (data.status === 'demo_mode') {
                this.updateModelStatus('Demo Mode', 'model-status-demo');
            } else if (data.status === 'success') {
                this.updateModelStatus('Model Loaded', 'model-status-loaded');
            } else {
                this.updateModelStatus('Error', 'model-status-error');
            }
        } catch (error) {
            console.error('Error checking model status:', error);
            this.updateModelStatus('Demo Mode', 'model-status-demo');
        }
    }

    updateModelStatus(text, className) {
        if (this.modelStatusEl) {
            this.modelStatusEl.textContent = text;
            this.modelStatusEl.className = `badge ${className}`;
        }
    }

    async sendMessage() {
        if (this.isLoading) return;

        const message = this.messageInput.value.trim();
        const category = this.categorySelect.value;

        if (!message) {
            this.showError('Please enter a message');
            return;
        }

        this.isLoading = true;
        this.updateSendButton(true);

        // Add user message to chat
        this.addMessage(message, 'user', category);

        // Clear input
        this.messageInput.value = '';
        this.categorySelect.value = '';

        // Show typing indicator
        const typingId = this.showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    category: category
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Remove typing indicator
                this.removeTypingIndicator(typingId);

                // Add AI response
                this.addMessage(data.response, 'ai', data.category, data.generation_time);

                // Update stats
                this.updatePerformanceStats(data.generation_time);

                // Show success feedback
                this.showSuccessFeedback();
            } else {
                this.removeTypingIndicator(typingId);
                this.showError(data.error || 'Failed to get response');
            }
        } catch (error) {
            this.removeTypingIndicator(typingId);
            this.showError('Network error: ' + error.message);
        } finally {
            this.isLoading = false;
            this.updateSendButton(false);
        }
    }

    addMessage(content, type, category = '', responseTime = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = type === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const messageParagraph = document.createElement('p');
        messageParagraph.textContent = content;

        const messageInfo = document.createElement('small');
        messageInfo.className = 'text-muted';

        if (type === 'user') {
            messageInfo.textContent = category ? `You (${category})` : 'You';
        } else {
            let infoText = 'AI Assistant';
            if (responseTime) {
                infoText += ` (${responseTime.toFixed(2)}s)`;
            }
            messageInfo.textContent = infoText;
        }

        messageContent.appendChild(messageParagraph);
        messageContent.appendChild(messageInfo);

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(messageContent);

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();

        // Animation
        messageDiv.style.opacity = '0';
        messageDiv.style.transform = 'translateY(20px)';
        setTimeout(() => {
            messageDiv.style.transition = 'all 0.3s ease';
            messageDiv.style.opacity = '1';
            messageDiv.style.transform = 'translateY(0)';
        }, 10);
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message ai-message';
        typingDiv.id = 'typing-indicator-' + Date.now();

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        const typingIndicator = document.createElement('div');
        typingIndicator.className = 'typing-indicator';
        typingIndicator.innerHTML = `
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            <span class="ms-2">AI is thinking...</span>
        `;

        messageContent.appendChild(typingIndicator);
        typingDiv.appendChild(avatar);
        typingDiv.appendChild(messageContent);

        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();

        return typingDiv.id;
    }

    removeTypingIndicator(id) {
        const typingDiv = document.getElementById(id);
        if (typingDiv) {
            typingDiv.remove();
        }
    }

    updateSendButton(loading) {
        if (loading) {
            this.sendButton.disabled = true;
            this.sendButton.innerHTML = '<div class="spinner-border spinner-border-sm" role="status"></div>';
        } else {
            this.sendButton.disabled = false;
            this.sendButton.innerHTML = '<i class="fas fa-paper-plane"></i>';
        }
    }

    updatePerformanceStats(responseTime) {
        this.conversationCount++;
        this.responseTimes.push(responseTime);

        // Update conversation count
        if (this.conversationCountEl) {
            this.conversationCountEl.textContent = this.conversationCount;
        }

        // Update average response time
        if (this.avgResponseTimeEl && this.responseTimes.length > 0) {
            const avgTime = this.responseTimes.reduce((a, b) => a + b, 0) / this.responseTimes.length;
            this.avgResponseTimeEl.textContent = avgTime.toFixed(2) + 's';
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    showError(message) {
        const toast = this.createToast('Error', message, 'danger');
        document.body.appendChild(toast);

        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    showSuccessFeedback() {
        // Subtle success animation
        this.sendButton.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.sendButton.style.transform = 'scale(1)';
        }, 100);
    }

    createToast(title, message, type = 'info') {
        const toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        toastContainer.style.zIndex = '9999';

        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-white bg-${type} border-0`;
        toast.setAttribute('role', 'alert');

        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <strong>${title}:</strong> ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" 
                        data-bs-dismiss="toast"></button>
            </div>
        `;

        toastContainer.appendChild(toast);
        return toastContainer;
    }

    // Utility methods
    formatTime(seconds) {
        return seconds < 1 ? `${(seconds * 1000).toFixed(0)}ms` : `${seconds.toFixed(2)}s`;
    }

    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showSuccessFeedback();
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.customerSupportApp = new CustomerSupportApp();
});

// Add some utility functions for analytics page
if (window.location.pathname.includes('analytics')) {
    // Auto-refresh analytics data every 30 seconds
    setInterval(() => {
        if (typeof loadAnalytics === 'function') {
            loadAnalytics();
        }
    }, 30000);
}

// Export for potential use in other scripts
window.CustomerSupportApp = CustomerSupportApp;