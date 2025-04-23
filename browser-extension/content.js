/**
 * Phishing URL Detection Browser Extension
 *
 * This browser extension analyzes all links on web pages to detect potential phishing URLs.
 * It communicates with a backend API to evaluate URLs and provides visual indicators to warn users.
 *
 * Key functionalities:
 * - Link scanning: The extension automatically detects and analyzes all links on web pages:
 *   - Processes links in batches to maintain performance.
 *   - Scans both initial page load and dynamically added content.
 *   - Tracks already analyzed URLs to prevent redundant API calls.
 * - URL normalization: The extension cleans and normalizes URLs before analysis:
 *   - Extracts actual URLs from redirect services (like Facebook's l.php).
 *   - Handles URLs with @ symbols that may be used in phishing attempts.
 *   - Removes tracking parameters (fbclid, utm_*).
 * - Visual indicators: The extension provides clear visual feedback about link safety:
 *   - Adds loading indicators during URL analysis.
 *   - Shows green indicators for safe URLs.
 *   - Shows red indicators for suspicious/phishing URLs.
 *   - Handles different types of page elements appropriately.
 * - Performance optimization: The extension uses various techniques to maintain browser performance:
 *   - Debouncing for DOM change detection.
 *   - Batch processing for link analysis.
 *   - Caching results to prevent redundant API calls.
 *   - Intelligent handling of single-page applications.
 *
 * Note:
 * - The extension communicates with a Flask backend API for URL analysis.
 * - Visual indicators are injected directly into the page DOM.
 * - The extension uses MutationObserver to detect dynamically loaded content.
 */

// Track URLs that have been fully analyzed to prevent overwrites
const analyzedUrls = new Map();

// Function to check if a URL is phishing using the API
async function checkUrl(url) {
    if (analyzedUrls.has(url)) {
        return analyzedUrls.get(url);
    }
    
    try {
        const apiUrl = 'https://debaiter-35538346792.asia-southeast1.run.app/predict';
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url: url }),
        });

        const data = await response.json();
        const result = data.result;
        const cleanedUrlFromAPI = data.cleaned_url || url;
        
        analyzedUrls.set(cleanedUrlFromAPI, result);
        
        return { result, cleanedUrl: cleanedUrlFromAPI };
    } catch (error) {
        console.error('Error checking URL:', error);
        return "unknown";
    }
}

// Function to add a loading indicator next to the link
function addLoadingIndicator(link) {
    if (link.getAttribute('data-indicator-added')) {
        if (link.getAttribute('data-indicator-status') !== 'loading') {
            return null;
        }
        
        const nextSibling = link.nextSibling;
        if (nextSibling && nextSibling.nodeType === Node.ELEMENT_NODE && 
            nextSibling.tagName === 'SPAN') {
            return nextSibling;
        }
    }

    const indicator = document.createElement('span');
    indicator.style.fontSize = '10px';
    indicator.style.marginLeft = '5px';
    indicator.style.padding = '2px 4px';
    indicator.style.borderRadius = '3px';
    indicator.style.color = 'white';
    indicator.style.backgroundColor = "#666";
    
    const parent = link.parentElement;

    if (parent && parent.getAttribute('data-ad-rendering-role') !== 'image') {
        indicator.textContent = "Analyzing...";
    } else {
        indicator.style.width = "8px";
        indicator.style.height = "8px";
        indicator.style.borderRadius = "50%";
        indicator.style.backgroundColor = "#666";
    }
    
    link.setAttribute('data-indicator-added', 'true');
    link.setAttribute('data-indicator-status', 'loading');

    if (link.parentElement) {
        link.parentElement.insertBefore(indicator, link.nextSibling);
    }
    
    return indicator;
}

// Function to update the indicator with results
function updateIndicator(link, indicator, status, cleanedUrl) {
    if (!indicator) {
        return;
    }
    
    const currentStatus = link.getAttribute('data-indicator-status');
    if (currentStatus && currentStatus !== 'loading') {
        if (currentStatus === 'safe' || currentStatus === 'phishing') {
            return;
        }
    }
    
    if (status === "phishing") {
        indicator.style.backgroundColor = "red";
    } else if (status === "safe") {
        indicator.style.backgroundColor = "green";
    } else {
        indicator.style.backgroundColor = "gray";
    }

    const parent = link.parentElement;
    if (parent && parent.getAttribute('data-ad-rendering-role') !== 'image') {
        indicator.textContent = status === "phishing" ? "Suspicious" : (status === "safe" ? "Safe" : "Unknown");
    }
    
    if (cleanedUrl && (status === 'safe' || status === 'phishing')) {
        link.setAttribute('data-checked-url', cleanedUrl);
    }
    
    link.setAttribute('data-indicator-status', status);
}

// Clean and normalize a URL
function normalizeUrl(url) {
    try {
        if (url.includes('l.facebook.com/l.php?u=')) {
            const decodedUrl = new URL(url);
            url = decodeURIComponent(decodedUrl.searchParams.get('u'));

            const extracted = decodeURIComponent(decodedUrl.searchParams.get('u') || '');

            if (!extracted || !extracted.includes('.')) {
                return { cleanedUrl: url, isFacebookLink: false, isObfuscated: true };
            }

            url = extracted;
        }

        if (url.includes('@')) {
            const splitUrl = url.split('@');
            if (splitUrl.length > 1) {
                url = `https://${splitUrl[1]}`;
            }
        }

        const urlObj = new URL(url);
        
        if (urlObj.hostname.endsWith('facebook.com') || urlObj.hostname.includes('.facebook.com')) {
            return { isFacebookLink: true };
        }

        const cleanUrl = new URL(url);
        cleanUrl.searchParams.forEach((_, key) => {
            if (key.startsWith('fbclid') || key.startsWith('utm_')) {
                cleanUrl.searchParams.delete(key);
            }
        });

        return { 
            cleanedUrl: cleanUrl.toString().split('?')[0].split('#')[0],
            isFacebookLink: false
        };
    } catch (error) {
        console.warn('Error normalizing URL:', url, error);
        return { cleanedUrl: url, isFacebookLink: false };
    }
}

// Process a batch of links
async function processBatch(links, batchSize = 5) {
    for (let i = 0; i < links.length; i += batchSize) {
        const batch = links.slice(i, i + batchSize);
        await Promise.all(batch.map(async (link) => {
            try {
                const currentStatus = link.getAttribute('data-indicator-status');
                if (currentStatus && currentStatus !== 'loading') {
                    return;
                }
                
                let url = link.href;

                if (!url || url === "" || url.startsWith('javascript:') || url === "#") {
                    return;
                }
                
                const checkedUrl = link.getAttribute('data-checked-url');
                if (checkedUrl === url) {
                    return;
                }

                let indicator = addLoadingIndicator(link);
                if (!indicator) return;
                
                const { cleanedUrl, isFacebookLink, isObfuscated } = normalizeUrl(url);

                if (isObfuscated) {
                    updateIndicator(link, indicator, "unknown", cleanedUrl);
                    return;
                }
                
                if (isFacebookLink) {
                    if (indicator && indicator.parentNode) {
                        indicator.parentNode.removeChild(indicator);
                    }
                    link.removeAttribute('data-indicator-added');
                    link.removeAttribute('data-indicator-status');
                    return;
                }

                const { result: status, cleanedUrl: apiCleanedUrl } = await checkUrl(cleanedUrl);

                updateIndicator(link, indicator, status, apiCleanedUrl);

            } catch (processError) {
                console.error('Error in link processing:', processError);
                const indicator = link.nextSibling;
                if (indicator && indicator.nodeType === Node.ELEMENT_NODE && indicator.tagName === 'SPAN') {
                    if (link.getAttribute('data-indicator-status') === 'loading') {
                        updateIndicator(link, indicator, "unknown");
                    }
                }
            }
        }));
        
        await new Promise(resolve => setTimeout(resolve, 300));
    }
}

// Function to scan all links on the page
async function scanLinks() {
    const links = [...document.querySelectorAll('a')];    
    await processBatch(links);
    
    console.log('Scan completed, processed', links.length, 'links');
}

// Initial scan on page load
setTimeout(() => {
    scanLinks();
}, 1000);

// Debounce function to limit API calls
function debounce(func, wait) {
    let timeout;
    return function() {
        const context = this;
        const args = arguments;
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            func.apply(context, args);
        }, wait);
    };
}

// Use MutationObserver to catch dynamically loaded content
const observer = new MutationObserver(debounce(() => {
    console.log('DOM changed, rescanning links');
    scanLinks();
}, 2000));

observer.observe(document.body, { childList: true, subtree: true });

// Add a listener for navigation events (for SPAs)
if (window.hasOwnProperty('onpopstate')) {
    window.addEventListener('popstate', () => {
        setTimeout(scanLinks, 1000);
    });
}