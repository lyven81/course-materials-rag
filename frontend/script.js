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
    
    // Ensure proper initial theme state after DOM is ready
    setTimeout(() => {
        const currentTheme = localStorage.getItem('theme') || 'light';
        updateThemeToggleState(currentTheme);
    }, 100);
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
    
    // Navigation button handlers
    const coursesBtn = document.querySelector('.nav-button.courses');
    const tryAskingBtn = document.querySelector('.nav-button.try-asking');
    
    if (coursesBtn) {
        coursesBtn.addEventListener('click', showCourseInfo);
    }
    
    if (tryAskingBtn) {
        tryAskingBtn.addEventListener('click', showSuggestedQuestions);
    }
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

// Navigation Functions
function showCourseInfo() {
    const courseStats = document.getElementById('courseStats');
    const totalCourses = document.getElementById('totalCourses').textContent;
    const courseTitles = document.getElementById('courseTitles').innerHTML;
    
    let courseList = 'No courses available';
    if (courseTitles && !courseTitles.includes('Loading') && !courseTitles.includes('Failed')) {
        const titles = courseTitles.split('</div>').map(title => 
            title.replace(/<[^>]*>/g, '').trim()
        ).filter(title => title);
        
        courseList = titles.length > 0 ? titles.join('\n• ') : 'No courses available';
        if (titles.length > 0) courseList = '• ' + courseList;
    }
    
    const message = `📚 **Course Information**\n\n**Total Courses:** ${totalCourses}\n\n**Available Courses:**\n${courseList}`;
    
    addMessage(message, 'assistant', null, null, false);
}

function showSuggestedQuestions() {
    const suggestedQuestions = [
        'What is the outline of the "MCP: Build Rich-Context AI Apps with Anthropic" course?',
        'Are there any courses that include a Chatbot implementation?', 
        'Are there any courses that explain what RAG is?',
        'What was covered in lesson 5 of the MCP course?'
    ];
    
    const questionList = suggestedQuestions.map(q => `• ${q}`).join('\n');
    const message = `💡 **Try asking these questions:**\n\n${questionList}\n\nSimply click on any question above or type your own!`;
    
    addMessage(message, 'assistant', null, null, false);
    
    // Add click handlers to the suggested questions in this message
    setTimeout(() => {
        const lastMessage = chatMessages.lastElementChild;
        if (lastMessage) {
            const messageContent = lastMessage.querySelector('.message-content');
            if (messageContent) {
                // Make the questions clickable
                suggestedQuestions.forEach(question => {
                    const questionRegex = new RegExp(`• ${question.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'g');
                    messageContent.innerHTML = messageContent.innerHTML.replace(
                        questionRegex,
                        `<span class="clickable-question" data-question="${question}">• ${question}</span>`
                    );
                });
                
                // Add click handlers
                messageContent.querySelectorAll('.clickable-question').forEach(span => {
                    span.style.cursor = 'pointer';
                    span.style.color = 'var(--primary-color)';
                    span.style.textDecoration = 'underline';
                    span.addEventListener('click', () => {
                        const question = span.getAttribute('data-question');
                        chatInput.value = question;
                        sendMessage();
                    });
                });
            }
        }
    }, 100);
}

// Theme Management Functions
function initializeTheme() {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme');
    
    // Default to light mode as specified in requirements
    const theme = savedTheme || 'light';
    setTheme(theme);
    
    // Listen for system theme changes only if no preference is saved
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
        if (!localStorage.getItem('theme')) {
            // Even with system preference, start with light mode by default
            // Users need to manually toggle to dark mode
            setTheme('light');
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
    // Apply theme
    if (theme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeToggle.setAttribute('aria-label', 'Switch to light mode');
        themeToggle.setAttribute('title', 'Switch to light mode');
    } else {
        document.documentElement.removeAttribute('data-theme');
        themeToggle.setAttribute('aria-label', 'Switch to dark mode');
        themeToggle.setAttribute('title', 'Switch to dark mode');
    }
    
    // Save theme preference to localStorage
    localStorage.setItem('theme', theme);
    
    // Update theme toggle button state
    updateThemeToggleState(theme);
}

function updateThemeToggleState(theme) {
    const sunIcon = themeToggle.querySelector('.sun-icon');
    const moonIcon = themeToggle.querySelector('.moon-icon');
    
    if (theme === 'dark') {
        // Dark mode: show sun icon (to switch to light)
        sunIcon.style.opacity = '1';
        sunIcon.style.transform = 'rotate(0deg) scale(1)';
        moonIcon.style.opacity = '0';
        moonIcon.style.transform = 'rotate(-180deg) scale(0)';
    } else {
        // Light mode: show moon icon (to switch to dark)
        sunIcon.style.opacity = '0';
        sunIcon.style.transform = 'rotate(180deg) scale(0)';
        moonIcon.style.opacity = '1';
        moonIcon.style.transform = 'rotate(0deg) scale(1)';
    }
}