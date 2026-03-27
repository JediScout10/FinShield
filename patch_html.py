import sys

def process_html():
    try:
        with open('static/index.html', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract the stitch code
        stitch_start = content.find('<!-- <!DOCTYPE html>')
        if stitch_start == -1:
            print('Could not find stitch start')
            return
            
        stitch_code = content[stitch_start:]
        # Remove the comment wrappers
        stitch_code = stitch_code.replace('<!-- <!DOCTYPE html>', '<!DOCTYPE html>')
        stitch_code = stitch_code.replace('</body></html> -->', '</body>\n</html>')
        
        # Add IDs to inputs
        # First input is Amount
        stitch_code = stitch_code.replace('placeholder="e.g. 1250.00" type="text"/>', 'id="txnAmount" placeholder="e.g. 1250.00" type="text"/>')
        # Second input is UID  
        stitch_code = stitch_code.replace('placeholder="USR-8829-X" type="text"/>', 'id="txnUid" placeholder="USR-8829-X" type="text"/>')
        
        # Add ID to button
        stitch_code = stitch_code.replace('Analyze Transaction\n                        </button>', 'Analyze Transaction\n                        </button>').replace('flex items-center justify-center gap-2">', 'flex items-center justify-center gap-2" id="analyzeBtn">')
        
        # Specific IDs for updateable DOM elements
        # 1. Probability Index
        stitch_code = stitch_code.replace('72.<span class="text-base opacity-60">45%</span>', '<span id="probValue">--.--%</span>')
        # 2. Risk Level Bar container
        stitch_code = stitch_code.replace('w-[72%]', 'w-[0%]" id="riskBar')
        # 3. Model Outcome Text
        stitch_code = stitch_code.replace('>FRAUD DETECTED<', ' id="modelOutcome">WAITING VERIFICATION<')
        # 4. Final Decision Text
        stitch_code = stitch_code.replace('Transaction Blocked\n                            </div>', 'Pending\n                            </div>').replace('<span class="material-symbols-outlined text-error">block</span>', '<span id="decisionIcon" class="material-symbols-outlined text-outline">hourglass_empty</span><span id="decisionText" class="ml-2">Pending</span>')
        
        # 5. High Risk badge
        stitch_code = stitch_code.replace('>High Risk<', ' id="riskBadge">Unknown<')
        
        # 6. Forensic trace replace the tbody content with an empty one with ID
        trace_start = stitch_code.find('<tbody class="divide-y divide-outline-variant/5">')
        trace_end = stitch_code.find('</tbody>', trace_start) + 8
        stitch_code = stitch_code[:trace_start] + '<tbody id="forensicTrace" class="divide-y divide-outline-variant/5"></tbody>' + stitch_code[trace_end:]
        
        # 7. Anomalous and Stability Markers ul IDs
        stitch_code = stitch_code.replace('<ul class="space-y-3">', '<ul id="anomalousSignals" class="space-y-3">', 1)
        stitch_code = stitch_code.replace('<ul class="space-y-3">', '<ul id="stabilityMarkers" class="space-y-3">', 1)

        # 8. Dynamic Device & IP Badges
        stitch_code = stitch_code.replace('<div class="text-xs text-primary font-mono bg-surface-container-highest p-2 rounded">OSX_13_4_SAFARI</div>', '<div id="deviceBadge" class="text-xs text-primary font-mono bg-surface-container-highest p-2 rounded">OSX_13_4_SAFARI</div>')
        stitch_code = stitch_code.replace('<div class="text-xs text-primary font-mono bg-surface-container-highest p-2 rounded">192.168.1.144</div>', '<div id="ipBadge" class="text-xs text-primary font-mono bg-surface-container-highest p-2 rounded">192.168.1.144</div>')
        
        # Javascript code to be injected
        js_code = """
<script>
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    
    // UI Loading state
    const btn = document.getElementById('analyzeBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="material-symbols-outlined text-sm animate-spin">refresh</span> Analyzing...';
    btn.disabled = true;

    const amount = document.getElementById('txnAmount').value;
    const uid = document.getElementById('txnUid').value || 'TEST_USER_1';
    
    if(!amount) {
        alert('Enter a valid amount');
        btn.innerHTML = originalText;
        btn.disabled = false;
        return;
    }
    
    // Generate simple device fingerprint dynamically as requested by user
    const ua = navigator.userAgent;
    let browser = "Unknown";
    if (ua.includes('Chrome')) browser = "Chrome";
    else if (ua.includes('Firefox')) browser = "Firefox";
    else if (ua.includes('Safari') && !ua.includes('Chrome')) browser = "Safari";
    else if (ua.includes('Edg')) browser = "Edge";
    
    let os = "Desktop";
    if (ua.includes('Win')) os = "Windows";
    else if (ua.includes('Mac')) os = "MacOS";
    else if (ua.includes('Linux')) os = "Linux";
    else if (ua.includes('Android')) os = "Android";
    else if (ua.includes('like Mac')) os = "iOS";
    
    const device = os + " | " + browser;
    document.getElementById('deviceBadge').innerText = device;
    
    try {
        const response = await fetch('/payment', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_id: uid,
                amount: parseFloat(amount),
                device_fingerprint: device
            })
        });
        
        const result = await response.json();
        
        // Update DOM Dynamic Elements
        if (result.detected_ip) {
            document.getElementById('ipBadge').innerText = result.detected_ip;
        }

        document.getElementById('probValue').innerHTML = result.probability + '%';
        const modelOutcomeEl = document.getElementById('modelOutcome');
        modelOutcomeEl.innerText = result.prediction.toUpperCase();

        if(result.prediction === 'Fraud') {
            modelOutcomeEl.className = 'text-4xl font-extrabold tracking-tighter text-error';
        } else if(result.prediction === 'Review Required') {
            modelOutcomeEl.className = 'text-4xl font-extrabold tracking-tighter text-orange-400';
        } else {
            modelOutcomeEl.className = 'text-4xl font-extrabold tracking-tighter text-tertiary';
        }
        
        document.getElementById('riskBar').style.width = result.probability + '%';
        
        document.getElementById('decisionText').innerText = result.decision;
        const icon = document.getElementById('decisionIcon');
        const badge = document.getElementById('riskBadge');
        badge.innerText = result.risk_level + ' Risk';
        
        if (result.decision === 'Blocked') {
            icon.innerText = 'block';
            icon.className = 'material-symbols-outlined text-error';
            badge.className = 'bg-error-container text-error px-3 py-1 text-[10px] font-black uppercase tracking-widest rounded-sm';
        } else if (result.decision === 'Approved') {
            icon.innerText = 'check_circle';
            icon.className = 'material-symbols-outlined text-tertiary';
            badge.className = 'bg-tertiary-container text-tertiary px-3 py-1 text-[10px] font-black uppercase tracking-widest rounded-sm';
        } else {
            icon.innerText = 'gavel';
            icon.className = 'material-symbols-outlined text-orange-400';
            badge.className = 'bg-orange-400/20 text-orange-400 px-3 py-1 text-[10px] font-black uppercase tracking-widest rounded-sm';
        }
        
        // Update trace
        const tbody = document.getElementById('forensicTrace');
        tbody.innerHTML = '';
        for (const [key, value] of Object.entries(result.feature_values)) {
            tbody.innerHTML += `<tr><td class="py-3 text-on-surface-variant">${key}</td><td class="py-3 font-mono">${value}</td><td class="py-3 text-right tabular text-tertiary">Extracted</td></tr>`;
        }
        
        // Update Anomalous Signals
        const anomUl = document.getElementById('anomalousSignals');
        anomUl.innerHTML = '';
        if(result.risk_factors.length === 0) anomUl.innerHTML = '<li class="text-xs opacity-50">None</li>';
        result.risk_factors.forEach(rf => {
            anomUl.innerHTML += `<li class="flex gap-3 items-start"><span class="material-symbols-outlined text-error text-lg">warning</span><span class="text-xs leading-tight">${rf}</span></li>`;
        });
        
        // Update Stability markers
        const stableUl = document.getElementById('stabilityMarkers');
        stableUl.innerHTML = '';
        if(result.safe_factors.length === 0) stableUl.innerHTML = '<li class="text-xs opacity-50">None</li>';
        result.safe_factors.forEach(sf => {
            stableUl.innerHTML += `<li class="flex gap-3 items-start"><span class="material-symbols-outlined text-tertiary text-lg">check_circle</span><span class="text-xs leading-tight">${sf}</span></li>`;
        });

        
    } catch(e) {
        console.error(e);
        alert('Error processing transaction');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});
</script>
"""
        stitch_code = stitch_code.replace('</body>', js_code + '\n</body>')
        
        with open('static/index.html', 'w', encoding='utf-8') as f:
            f.write(stitch_code)
        
        print('HTML successfully extracted and patched.')
    except Exception as e:
        print(f"Exception happened: {e}")

process_html()
