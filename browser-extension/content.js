// Dummy function to simulate detection API (for now, it just marks all URLs as "safe")
function checkUrl(url) {
    return "safe"; // Marks all URLs as "safe" for now
}

// Function to add an indicator next to the link
function addIndicator(link, status) {
    if (link.getAttribute('data-indicator-added')) {
        return; // Prevent duplicate indicators
    }

    const indicator = document.createElement('span');
    indicator.style.fontSize = '10px';
    indicator.style.marginLeft = '5px';
    indicator.style.padding = '2px 4px';
    indicator.style.borderRadius = '3px';
    indicator.style.color = 'white';
    indicator.style.backgroundColor = status === "phishing" ? "red" : "green";

    // Check if the parent has data-ad-rendering-role="image" and skip adding textContent if true
    const parent = link.parentElement;
    if (parent && parent.getAttribute('data-ad-rendering-role') !== 'image') {
        indicator.textContent = status === "phishing" ? "Phishing" : "Safe";
    }
    
    link.setAttribute('data-indicator-added', 'true');
    link.parentElement.insertBefore(indicator, link.nextSibling);
}

// Function to scan all links on the page
function scanLinks() {
    // Extract and process links
    const links = [...document.querySelectorAll('a')];
    const extractedUrls = links.map(link => {
        let url = link.href;
        let cleanedUrl = null;

        try {
            // Extract actual URL if it's a Facebook tracking link
            if (url.includes('l.facebook.com/l.php?u=')) {
                const decodedUrl = new URL(url);
                url = decodeURIComponent(decodedUrl.searchParams.get('u')); // Extract actual URL
            }

            // Check AFTER extracting actual URL
            const urlObj = new URL(url);
            if (urlObj.hostname.endsWith('facebook.com') || urlObj.hostname.includes('.facebook.com')) {
                return null; // Skip Facebook links
            }

            // Handle URLs with '@' (e.g., user@example.com@evil.com)
            if (url.includes('@')) {
                const splitUrl = url.split('@');
                if (splitUrl.length > 1) {
                    url = `https://${splitUrl[1]}`; // Extract the part after '@'
                }
            }

            // Remove tracking parameters
            const cleanUrl = new URL(url);
            cleanUrl.searchParams.forEach((_, key) => {
                if (key.startsWith('fbclid') || key.startsWith('utm_')) {
                    cleanUrl.searchParams.delete(key);
                }
            });

            cleanedUrl = cleanUrl.toString().split('?')[0]; // Remove any remaining query params

            // Determine if it's phishing (replace with real check)
            const status = checkUrl(cleanedUrl);
            addIndicator(link, status); // Add the indicator next to the link

        } catch (error) {
            console.warn('Error processing URL:', url, error);
            return null;
        }

        return cleanedUrl;
    }).filter(Boolean); // Remove null/undefined entries

    console.log('Extracted URLs:', extractedUrls);
}

// Initial scan on page load
scanLinks();

// Use MutationObserver to catch dynamically loaded content
const observer = new MutationObserver((mutations) => {
    mutations.forEach(() => {
        scanLinks();
    });
});

observer.observe(document.body, { childList: true, subtree: true });
