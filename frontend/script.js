// API base URL - use relative path to work from any host
const API_URL = '/api';

// Global state
let currentSessionId = null;

// DOM elements
let chatMessages, chatInput, sendButton, totalCourses, courseTitles, newChatButton, themeToggle;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Get DOM elements after page loads
    chatMessages = document.getElementById('chatMessages');
    chatInput = document.getElementById('chatInput');
    sendButton = document.getElementById('sendButton');
    totalCourses = document.getElementById('totalCourses');
    courseTitles = document.getElementById('courseTitles');
    newChatButton = document.getElementById('newChatButton');
    themeToggle = document.getElementById('themeToggle');
    
    setupEventListeners();
    initializeTheme();
    createNewSession();
    loadCourseStats();
});

// Event Listeners
function setupEventListeners() {
    // Chat functionality
    sendButton.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
    
    // New chat functionality
    newChatButton.addEventListener('click', createNewSession);
    
    // Theme toggle functionality
    themeToggle.addEventListener('click', toggleTheme);
    themeToggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleTheme();
        }
    });
    
    // Suggested questions
    document.querySelectorAll('.suggested-item').forEach(button => {
        button.addEventListener('click', (e) => {
            const question = e.target.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });
}


// Chat Functions
async function sendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;

    // Disable input
    chatInput.value = '';
    chatInput.disabled = true;
    sendButton.disabled = true;

    // Add user message
    addMessage(query, 'user');

    // Add loading message - create a unique container for it
    const loadingMessage = createLoadingMessage();
    chatMessages.appendChild(loadingMessage);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: currentSessionId
            })
        });

        let data;
        if (!response.ok) {
            // Try to get error details from response
            try {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Request failed with status ${response.status}`);
            } catch (jsonError) {
                throw new Error(`Request failed with status ${response.status}`);
            }
        }

        data = await response.json();
        
        // Update session ID if new
        if (!currentSessionId) {
            currentSessionId = data.session_id;
        }

        // Replace loading message with response
        loadingMessage.remove();
        addMessage(data.answer, 'assistant', data.sources, data.source_summary);

    } catch (error) {
        // Replace loading message with error
        loadingMessage.remove();
        
        // Provide helpful error messages based on error type
        let errorMessage = error.message;
        if (error.message.includes('overloaded') || error.message.includes('high demand')) {
            errorMessage = '⚠️ The AI service is currently experiencing high demand. Please try again in a few minutes.';
        } else if (error.message.includes('rate limit')) {
            errorMessage = '⏳ Please wait a moment before sending another message.';
        } else if (error.message.includes('connection') || error.message.includes('Failed to fetch')) {
            errorMessage = '🔌 Connection issue. Please check your internet connection and try again.';
        } else if (error.message.includes('temporarily unavailable')) {
            errorMessage = '🔧 AI service is temporarily unavailable. Please try again later.';
        } else if (!error.message || error.message === 'Query failed') {
            errorMessage = '❌ Something went wrong. Please try again.';
        }
        
        addMessage(errorMessage, 'assistant');
    } finally {
        chatInput.disabled = false;
        sendButton.disabled = false;
        chatInput.focus();
    }
}

function createLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="loading">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    return messageDiv;
}

function addMessage(content, type, sources = null, sourceSummary = null, isWelcome = false) {
    const messageId = Date.now();
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}${isWelcome ? ' welcome-message' : ''}`;
    messageDiv.id = `message-${messageId}`;
    
    // Convert markdown to HTML for assistant messages
    let displayContent = type === 'assistant' ? marked.parse(content) : escapeHtml(content);
    
    // Add inline citations if sources are available
    if (sources && sources.length > 0 && type === 'assistant') {
        displayContent = addInlineCitations(displayContent, sources);
    }
    
    let html = `<div class="message-content">${displayContent}</div>`;
    
    // Add source summary if available
    if (sourceSummary) {
        html += `<div class="source-summary">${sourceSummary}</div>`;
    }
    
    if (sources && sources.length > 0) {
        html += createSourcesSection(sources);
    }
    
    messageDiv.innerHTML = html;
    
    // Add click handlers for inline citations
    if (sources && sources.length > 0) {
        addCitationClickHandlers(messageDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageId;
}

function addInlineCitations(content, sources) {
    // This is a simple implementation that adds citations at the end of sentences
    // In a more sophisticated version, you could use NLP to match content to sources
    
    if (!sources || sources.length === 0) return content;
    
    // Add citations to sentences that end with periods, but avoid citations in code blocks
    // Simple heuristic: add citations after sentences that seem to contain factual information
    const sentences = content.split('.');
    let citationIndex = 0;
    
    return sentences.map((sentence, index) => {
        if (index === sentences.length - 1) return sentence; // Don't modify the last fragment
        
        // Skip if sentence is too short or is likely in a code block
        if (sentence.trim().length < 20 || sentence.includes('<code>') || sentence.includes('```')) {
            return sentence + '.';
        }
        
        // Add citation if we still have sources to cite
        if (citationIndex < sources.length) {
            const citationId = sources[citationIndex].citation_id;
            citationIndex++;
            return sentence + `<span class="inline-citation" data-citation-id="${citationId}">[${citationId}]</span>.`;
        }
        
        return sentence + '.';
    }).join('');
}

function createSourcesSection(sources) {
    // Create rich source cards instead of simple text
    let sourcesHtml = `
        <details class="sources-collapsible">
            <summary class="sources-header">Sources (${sources.length})</summary>
            <div class="sources-container">
    `;
    
    sources.forEach((source, index) => {
        const confidenceClass = getConfidenceClass(source.relevance_score);
        const lessonText = source.lesson_number ? `Lesson ${source.lesson_number}` : '';
        const lessonTitle = source.lesson_title ? `: ${source.lesson_title}` : '';
        
        sourcesHtml += `
            <div class="source-card ${confidenceClass}" data-citation-id="${source.citation_id}">
                <div class="source-header">
                    <div class="source-citation-number">[${source.citation_id}]</div>
                    <div class="source-info">
                        <div class="source-course-title">
                            ${source.course_link ? 
                                `<a href="${source.course_link}" target="_blank">${source.course_title}</a>` : 
                                source.course_title
                            }
                        </div>
                        ${lessonText || lessonTitle ? 
                            `<div class="source-lesson-info">
                                ${source.lesson_link ? 
                                    `<a href="${source.lesson_link}" target="_blank">${lessonText}${lessonTitle}</a>` :
                                    `${lessonText}${lessonTitle}`
                                }
                            </div>` : ''
                        }
                    </div>
                    <div class="source-confidence" title="Relevance: ${Math.round(source.relevance_score * 100)}%">
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${source.relevance_score * 100}%"></div>
                        </div>
                    </div>
                </div>
                <details class="source-preview">
                    <summary class="source-preview-toggle">Show preview</summary>
                    <div class="source-snippet">${source.content_snippet}</div>
                </details>
            </div>
        `;
    });
    
    sourcesHtml += `
            </div>
        </details>
    `;
    
    return sourcesHtml;
}

function getConfidenceClass(relevanceScore) {
    if (relevanceScore >= 0.8) return 'high-confidence';
    if (relevanceScore >= 0.6) return 'medium-confidence';
    return 'low-confidence';
}

function addCitationClickHandlers(messageDiv) {
    // Add click handlers to inline citations
    const inlineCitations = messageDiv.querySelectorAll('.inline-citation');
    inlineCitations.forEach(citation => {
        citation.addEventListener('click', (e) => {
            e.preventDefault();
            const citationId = citation.getAttribute('data-citation-id');
            highlightSourceCard(messageDiv, citationId);
            
            // Open the sources section if it's closed
            const sourcesCollapsible = messageDiv.querySelector('.sources-collapsible');
            if (sourcesCollapsible && !sourcesCollapsible.open) {
                sourcesCollapsible.open = true;
            }
        });
    });
    
    // Add hover handlers for source cards
    const sourceCards = messageDiv.querySelectorAll('.source-card');
    sourceCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            const citationId = card.getAttribute('data-citation-id');
            highlightInlineCitation(messageDiv, citationId);
        });
        
        card.addEventListener('mouseleave', () => {
            clearInlineCitationHighlights(messageDiv);
        });
    });
}

function highlightSourceCard(messageDiv, citationId) {
    // Clear existing highlights
    const allCards = messageDiv.querySelectorAll('.source-card');
    allCards.forEach(card => card.classList.remove('highlighted'));
    
    // Highlight the target card
    const targetCard = messageDiv.querySelector(`[data-citation-id="${citationId}"]`);
    if (targetCard) {
        targetCard.classList.add('highlighted');
        targetCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
        // Remove highlight after 2 seconds
        setTimeout(() => {
            targetCard.classList.remove('highlighted');
        }, 2000);
    }
}

function highlightInlineCitation(messageDiv, citationId) {
    const citation = messageDiv.querySelector(`.inline-citation[data-citation-id="${citationId}"]`);
    if (citation) {
        citation.classList.add('highlighted');
    }
}

function clearInlineCitationHighlights(messageDiv) {
    const citations = messageDiv.querySelectorAll('.inline-citation.highlighted');
    citations.forEach(citation => citation.classList.remove('highlighted'));
}

// Helper function to escape HTML for user messages
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Removed removeMessage function - no longer needed since we handle loading differently

async function createNewSession() {
    // Clear current session state
    currentSessionId = null;
    
    // Clear chat messages
    chatMessages.innerHTML = '';
    
    // Clear input field 
    chatInput.value = '';
    
    // Re-enable input in case it was disabled
    chatInput.disabled = false;
    sendButton.disabled = false;
    
    // Add welcome message
    addMessage('Welcome to the Course Materials Assistant! I can help you with questions about courses, lessons and specific content. What would you like to know?', 'assistant', null, null, true);
    
    // Focus input for immediate typing
    chatInput.focus();
}

// Load course statistics
async function loadCourseStats() {
    try {
        console.log('Loading course stats...');
        const response = await fetch(`${API_URL}/courses`);
        if (!response.ok) throw new Error('Failed to load course stats');
        
        const data = await response.json();
        console.log('Course data received:', data);
        
        // Update stats in UI
        if (totalCourses) {
            totalCourses.textContent = data.total_courses;
        }
        
        // Update course titles
        if (courseTitles) {
            if (data.course_titles && data.course_titles.length > 0) {
                courseTitles.innerHTML = data.course_titles
                    .map(title => `<div class="course-title-item">${title}</div>`)
                    .join('');
            } else {
                courseTitles.innerHTML = '<span class="no-courses">No courses available</span>';
            }
        }
        
    } catch (error) {
        console.error('Error loading course stats:', error);
        // Set default values on error
        if (totalCourses) {
            totalCourses.textContent = '0';
        }
        if (courseTitles) {
            courseTitles.innerHTML = '<span class="error">Failed to load courses</span>';
        }
    }
}

// Theme Management Functions
function initializeTheme() {
    // Check for saved theme preference or default to system preference
    const savedTheme = localStorage.getItem('theme');
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    const theme = savedTheme || (systemPrefersDark ? 'dark' : 'light');
    setTheme(theme);
    
    // Listen for system theme changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            setTheme(e.matches ? 'dark' : 'light');
        }
    });
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    
    // Add a subtle animation feedback
    themeToggle.style.transform = 'scale(0.9)';
    setTimeout(() => {
        themeToggle.style.transform = '';
    }, 150);
}

function setTheme(theme) {
    if (theme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        themeToggle.setAttribute('aria-label', 'Switch to dark mode');
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeToggle.setAttribute('aria-label', 'Switch to light mode');
    }
    
    // Save theme preference
    localStorage.setItem('theme', theme);
}