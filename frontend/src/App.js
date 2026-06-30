import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [cameraActive, setCameraActive] = useState(false);
  const [wordsList, setWordsList] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [wordLog, setWordLog] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [isSigning, setIsSigning] = useState(false);
  const [selectedWord, setSelectedWord] = useState(null);
  const [dictionaryOpen, setDictionaryOpen] = useState(true);
  const [leftHandActive, setLeftHandActive] = useState(false);
  const [rightHandActive, setRightHandActive] = useState(false);
  const [poseStatus, setPoseStatus] = useState('NOT_DETECTED');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.40);
  const [speechSpeed, setSpeechSpeed] = useState(1.0);
  const [isMuted, setIsMuted] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const wordLogRef = React.useRef(wordLog);
  const isMutedRef = React.useRef(isMuted);
  const confidenceThresholdRef = React.useRef(confidenceThreshold);

  useEffect(() => {
    wordLogRef.current = wordLog;
  }, [wordLog]);

  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  useEffect(() => {
    confidenceThresholdRef.current = confidenceThreshold;
  }, [confidenceThreshold]);

  const getBackendUrl = (path) => {
    if (window.location.port === '3000') {
      return `http://127.0.0.1:5000${path}`;
    }
    return path;
  };

  useEffect(() => {
    fetchWords();
  }, [currentPage]);

  useEffect(() => {
    let intervalId = null;

    if (cameraActive && currentPage === 'app') {
      intervalId = setInterval(() => {
        pollPredictions();
      }, 200);
    } else {
      setPredictions([]);
      setIsSigning(false);
      setLeftHandActive(false);
      setRightHandActive(false);
      setPoseStatus('NOT_DETECTED');
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [cameraActive, currentPage]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (currentPage !== 'app') return;
      if (e.code === 'Space') {
        e.preventDefault();
        toggleCamera();
      } else if (e.code === 'Escape') {
        clearLog();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [cameraActive, currentPage]);

  const fetchWords = async () => {
    try {
      const res = await fetch(getBackendUrl('/api/words'));
      if (res.ok) {
        const data = await res.json();
        if (Array.isArray(data)) {
          setWordsList(data);
        } else if (data && data.high && Array.isArray(data.high)) {
          setWordsList(data.high);
        } else {
          setWordsList([]);
        }
      }
    } catch (err) {
      console.error("Failed to load dictionary list:", err);
    }
  };

  const pollPredictions = async () => {
    try {
      const res = await fetch(getBackendUrl('/api/predictions'));
      if (res.ok) {
        const data = await res.json();
        setPredictions(data.top_predictions || []);
        
        const rawLog = data.word_log || [];
        if (rawLog.length > 0) {
          const lastWord = rawLog[rawLog.length - 1];
          const bestPred = data.top_predictions?.[0];
          if (bestPred && bestPred.word === lastWord && bestPred.confidence >= confidenceThresholdRef.current) {
            if (wordLogRef.current.length < rawLog.length) {
              setWordLog(rawLog);
              if (!isMutedRef.current) {
                speakText(lastWord);
              }
            }
          } else {
            setWordLog(rawLog);
          }
        } else {
          setWordLog([]);
        }
        
        setIsSigning(data.is_signing || false);
        setLeftHandActive(data.left_hand_active || false);
        setRightHandActive(data.right_hand_active || false);
        setPoseStatus(data.pose_status || 'NOT_DETECTED');
      }
    } catch (err) {
      console.warn("Prediction polling failed:", err);
    }
  };

  const toggleCamera = async () => {
    const endpoint = cameraActive ? '/api/camera/stop' : '/api/camera/start';
    try {
      const res = await fetch(getBackendUrl(endpoint), { method: 'POST' });
      if (res.ok) {
        setCameraActive(!cameraActive);
      }
    } catch (err) {
      console.error("Failed to toggle camera:", err);
    }
  };

  const toggleMirror = async () => {
    try {
      await fetch(getBackendUrl('/api/toggle_mirror'), { method: 'POST' });
    } catch (err) {
      console.error("Failed to toggle mirror state:", err);
    }
  };

  const clearLog = async () => {
    try {
      const res = await fetch(getBackendUrl('/api/clear'), { method: 'POST' });
      if (res.ok) {
        setWordLog([]);
        setPredictions([]);
      }
    } catch (err) {
      console.error("Failed to clear log:", err);
    }
  };

  const handleConfidenceChange = async (e) => {
    const val = parseFloat(e.target.value);
    setConfidenceThreshold(val);
    try {
      await fetch(getBackendUrl('/api/settings'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ confidence_threshold: val })
      });
    } catch (err) {
      console.error("Failed to update confidence threshold on backend:", err);
    }
  };

  const exitToLanding = async () => {
    if (cameraActive) {
      await toggleCamera();
    }
    setCurrentPage('landing');
  };

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = speechSpeed;
      window.speechSynthesis.speak(utterance);
    }
  };

  const speakFullSentence = () => {
    if (wordLog.length > 0) {
      speakText(wordLog.join(' '));
    }
  };

  const exportSession = () => {
    if (wordLog.length === 0) return;
    const content = `SignSight Translation Log\nDate: ${new Date().toLocaleString()}\n\nRecognized Text:\n${wordLog.join(' ')}\n\nWord Chips History: ${JSON.stringify(wordLog, null, 2)}`;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `signsight-session-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const filteredWords = Array.isArray(wordsList)
    ? wordsList.filter(word => typeof word === 'string' && word.toLowerCase().includes(searchQuery.toLowerCase()))
    : [];

  if (currentPage === 'landing') {
    return (
      <div className="landing-container">
        <header className="landing-header">
          <div className="logo-area">
            <i className="fa-solid fa-hands-asl-interpreting logo-icon"></i>
            <h1>SignSight <span>V1</span></h1>
          </div>
          <button className="cta-btn" onClick={() => setCurrentPage('app')}>
            Launch Detector <i className="fa-solid fa-arrow-right"></i>
          </button>
        </header>

        <section className="hero-section">
          <div className="hero-content">
            <span className="hero-badge">Full-Stack Real-Time Sign Language Translation</span>
            <h2>Bridging Communication through Real-Time ASL Interpretation</h2>
            <p>
              SignSight leverages computer vision and deep learning sequence models to capture and translate 
              American Sign Language (ASL) gestures live from your camera. Simply sit back, sign, and translate.
            </p>
            <div className="hero-actions">
              <button className="primary-btn" onClick={() => setCurrentPage('app')}>
                Start Translation <i className="fa-solid fa-wand-magic-sparkles"></i>
              </button>
            </div>
          </div>
          
          <div className="hero-image">
            <div className="glowing-orb"></div>
            <div className="stats-grid">
              <div className="metric-card">
                <h3>77 Words</h3>
                <p>Wide ASL Vocabulary</p>
              </div>
              <div className="metric-card">
                <h3>74.26%</h3>
                <p>Top-1 Validation Accuracy</p>
              </div>
              <div className="metric-card">
                <h3>94.88%</h3>
                <p>Top-5 Validation Accuracy</p>
              </div>
              <div className="metric-card">
                <h3>60 FPS</h3>
                <p>Real-Time Extraction</p>
              </div>
            </div>
          </div>
        </section>

        <section className="features-section">
          <h3>Interactive System Modules</h3>
          <div className="features-grid">
            <div className="feature-card">
              <i className="fa-solid fa-hand feature-icon"></i>
              <h4>MediaPipe Tracking</h4>
              <p>Exposes real-time upper skeleton lines and 21-joint hand coordinate arrays with zero latency.</p>
            </div>
            <div className="feature-card">
              <i className="fa-solid fa-chart-line feature-icon"></i>
              <h4>BiLSTM Prediction</h4>
              <p>Classifies motion over time, filtering speed, velocity, and knuckle distances to compute translations.</p>
            </div>
            <div className="feature-card">
              <i className="fa-solid fa-circle-question feature-icon"></i>
              <h4>Pose Diagnostics</h4>
              <p>Tracks your position relative to the camera to give you active feedback and instructions.</p>
            </div>
          </div>
        </section>

        <section className="references-section">
          <h3>Dataset Credits & Academic References</h3>
          <p className="section-intro">This project is built using models trained on benchmark public sign language databases. We appreciate and attribute the authors for their open contributions to academic research:</p>
          <div className="references-grid">
            <div className="reference-card">
              <div className="ref-badge">ASL Citizen</div>
              <h4>ASL Citizen Dataset (Microsoft Research)</h4>
              <p className="citation-text">
                <strong>Citation:</strong> Sundararaman et al. "ASL Citizen: A Community-Sourced Dataset for Word-Level American Sign Language Recognition." 
                Crowdsourced high-definition video assets covering extensive vocabulary variations.
              </p>
              <a href="https://www.microsoft.com/en-us/research/project/asl-citizen/" target="_blank" rel="noreferrer" className="ref-link">Visit Microsoft Project <i className="fa-solid fa-arrow-up-right-from-square"></i></a>
            </div>
            
            <div className="reference-card">
              <div className="ref-badge">WLASL (WASL)</div>
              <h4>WLASL Dataset (Kaggle)</h4>
              <p className="citation-text">
                <strong>Citation:</strong> Li et al. "Word-Level Deep Sign Language Recognition from Video." 
                The reference database for word-level American Sign Language recognition, sourced from the public Kaggle database.
              </p>
              <a href="https://www.kaggle.com/datasets/david1013/wlasl-dataset" target="_blank" rel="noreferrer" className="ref-link">Visit Kaggle Dataset <i className="fa-solid fa-arrow-up-right-from-square"></i></a>
            </div>
          </div>
        </section>

        <footer className="landing-footer">
          <p>© 2026 SignSight V1</p>
        </footer>
      </div>
    );
  }

  return (
    <div className={`app-container app-layout-2col ${dictionaryOpen ? 'dict-open' : 'dict-closed'}`}>
      <main className="main-content">
        <header className="top-header dashboard-header">
          <div className="header-left">
            <button className="control-btn back-btn" onClick={exitToLanding}>
              <i className="fa-solid fa-arrow-left"></i> Back
            </button>
            <div className="logo-area clickable" onClick={exitToLanding}>
              <i className="fa-solid fa-hands-asl-interpreting logo-icon"></i>
              <h2>SignSight <span>V1</span></h2>
            </div>
          </div>

          <div className="header-stats">
            <span className="header-stat">🏆 Top-1: 74.26%</span>
            <span className="header-stat">⭐ Top-5: 94.88%</span>
            <span className="header-stat">📚 Vocab: 77 Signs</span>
            <button className={`control-btn dict-toggle-btn ${dictionaryOpen ? 'active' : ''}`} onClick={() => setDictionaryOpen(!dictionaryOpen)}>
              <i className={`fa-solid ${dictionaryOpen ? 'fa-book-open' : 'fa-book'}`}></i>
              {dictionaryOpen ? 'Hide Dictionary' : 'Show Dictionary'}
            </button>
            <button className="settings-toggle-btn" onClick={() => setShowSettings(true)}>
              <i className="fa-solid fa-sliders"></i> Settings
            </button>
          </div>
        </header>

        <div className={`settings-overlay-backdrop ${showSettings ? 'active' : ''}`} onClick={() => setShowSettings(false)}>
          <div className="settings-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <h3>System Configurations</h3>
              <button className="close-drawer-btn" onClick={() => setShowSettings(false)}>&times;</button>
            </div>
            
            <div className="drawer-body">
              <div className="setting-row">
                <div className="setting-label-row">
                  <label>Confidence Log Threshold</label>
                  <span className="setting-val-badge">{Math.round(confidenceThreshold * 100)}%</span>
                </div>
                <input 
                  type="range" 
                  min="0.20" 
                  max="0.90" 
                  step="0.05" 
                  value={confidenceThreshold} 
                  onChange={handleConfidenceChange}
                />
                <p className="setting-desc">Signs with inference score below this will be shown on dashboard but skipped from translation logs.</p>
              </div>

              <div className="setting-row">
                <div className="setting-label-row">
                  <label>Narration Speech Rate</label>
                  <span className="setting-val-badge">{speechSpeed}x</span>
                </div>
                <input 
                  type="range" 
                  min="0.5" 
                  max="2.0" 
                  step="0.1" 
                  value={speechSpeed} 
                  onChange={(e) => setSpeechSpeed(parseFloat(e.target.value))}
                />
              </div>

              <div className="setting-row toggle-row">
                <label>Automatic Voice Readout</label>
                <button 
                  className={`toggle-btn ${!isMuted ? 'active' : ''}`}
                  onClick={() => setIsMuted(!isMuted)}
                >
                  {!isMuted ? 'Narrator Active' : 'Narrator Muted'}
                </button>
              </div>

              <div className="shortcut-guide">
                <h4>System Keyboard Hotkeys</h4>
                <p><kbd>Space</kbd> Toggle camera hardware</p>
                <p><kbd>Esc</kbd> Reset session word chips</p>
              </div>
            </div>
          </div>
        </div>

        <div className="grid-layout">
          <div className={`card video-card ${cameraActive ? 'feed-active' : ''} ${isSigning ? 'border-pulse' : ''}`}>
            <div className="video-container">
              <img
                id="video-feed"
                src={getBackendUrl(`/video_feed?status=${cameraActive ? 'active' : 'inactive'}`)}
                alt="SignSight Webcam Feed"
              />

              <div className="state-overlay">
                <span className={`status-indicator ${isSigning ? 'recording' : 'waiting'}`}>
                  <span className="dot"></span>
                  <span className="status-text">
                    {isSigning ? 'RECORDING SIGN' : cameraActive ? 'SYSTEM READY' : 'STANDBY'}
                  </span>
                </span>
              </div>
            </div>

            {cameraActive && (
              <div className="diagnostics-panel">
                <div className={`diag-item ${leftHandActive ? 'active' : ''}`}>
                  <span className="diag-dot"></span>
                  <span>Left Hand: {leftHandActive ? 'Active' : 'Missing'}</span>
                </div>
                <div className={`diag-item ${rightHandActive ? 'active' : ''}`}>
                  <span className="diag-dot"></span>
                  <span>Right Hand: {rightHandActive ? 'Active' : 'Missing'}</span>
                </div>
                <div className={`diag-pose ${poseStatus}`}>
                  {poseStatus === 'TOO_CLOSE' && '⚠️ Please sit further back'}
                  {poseStatus === 'NOT_DETECTED' && '🔍 Searching for upper body...'}
                  {poseStatus === 'OK' && '✅ Optimal distance'}
                  {poseStatus === 'TOO_FAST' && '⚠️ Sign too quick! Please sign slower and try again.'}
                </div>
              </div>
            )}

            <div className="video-controls">
              <button
                id="btn-camera"
                className={`control-btn ${cameraActive ? 'active-camera' : ''}`}
                onClick={toggleCamera}
              >
                <i className={`fa-solid ${cameraActive ? 'fa-video-slash' : 'fa-video'}`}></i>
                {cameraActive ? 'Stop Camera' : 'Start Camera'}
              </button>
              <button id="btn-mirror" className="control-btn" onClick={toggleMirror}>
                <i className="fa-solid fa-arrows-left-right"></i> Toggle Mirror
              </button>
              <button id="btn-clear" className="control-btn danger" onClick={clearLog}>
                <i className="fa-solid fa-trash-can"></i> Clear Log
              </button>
            </div>
          </div>

          <div className="card predictions-card">
            <div className="card-header-actions">
              <h3>Translation Dashboard</h3>
              <div className="action-buttons">
                <button 
                  className={`icon-action-btn ${isMuted ? 'muted-active' : ''}`} 
                  onClick={() => setIsMuted(!isMuted)} 
                  title={isMuted ? "Unmute text-to-speech" : "Mute text-to-speech"}
                >
                  <i className={`fa-solid ${isMuted ? 'fa-volume-xmark' : 'fa-volume-high'}`}></i>
                </button>
                <button 
                  className="icon-action-btn" 
                  onClick={speakFullSentence} 
                  title="Speak log out loud" 
                  disabled={wordLog.length === 0}
                >
                  <i className="fa-solid fa-comment-dots"></i>
                </button>
                <button 
                  className="icon-action-btn" 
                  onClick={exportSession} 
                  title="Export translation log" 
                  disabled={wordLog.length === 0}
                >
                  <i className="fa-solid fa-file-export"></i>
                </button>
              </div>
            </div>

            <div className="predictions-list" id="preds-container">
              {predictions.length > 0 ? (
                predictions.map((p, idx) => (
                  <div key={idx} className="pred-row">
                    <div className="pred-labels">
                      <span className="pred-word">{p.word}</span>
                      <span className="pred-conf">{(p.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bar-container">
                      <div className="bar-fill" style={{ width: `${p.confidence * 100}%` }}></div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="empty-predictions">
                  {cameraActive ? (
                    <>
                      <i className="fa-solid fa-spinner fa-spin"></i>
                      <p>Perform a sign gesture in frame...</p>
                    </>
                  ) : (
                    <>
                      <i className="fa-solid fa-circle-play"></i>
                      <p>Press "Start Camera" to initialize the model feed.</p>
                    </>
                  )}
                </div>
              )}
            </div>

            <div className="recognized-log-box">
              <span className="section-label">Recognized Sentence Log</span>
              <div className="word-chips" id="recognized-words">
                {wordLog.length > 0 ? (
                  wordLog.map((word, idx) => {
                    return (
                      <span key={idx} className="word-chip">{word}</span>
                    );
                  })
                ) : (
                  <span className="empty-chip">No signs captured yet. Try a gesture below!</span>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <section className="dictionary-panel">
        <div className="dictionary-header">
          <h3><i className="fa-solid fa-book-open"></i> Reference Dictionary</h3>
          <p>Watch reference clips to learn and mimic gestures</p>
          <div className="search-bar-container">
            <i className="fa-solid fa-magnifying-glass search-icon"></i>
            <input 
              type="text" 
              placeholder="Search 77 words..." 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="dictionary-search-input"
            />
          </div>
        </div>

        <div className="dictionary-words-scroll">
          <div className="word-grid transition-grid">
            {filteredWords.length > 0 ? (
              filteredWords.map((word) => {
                const isExpanded = selectedWord === word;
                return (
                  <div 
                    key={word} 
                    className={`word-card-wrapper ${isExpanded ? 'expanded' : ''}`}
                  >
                    <button
                      className={`word-btn ${isExpanded ? 'active' : ''}`}
                      onClick={() => setSelectedWord(isExpanded ? null : word)}
                    >
                      <span>{word}</span>
                      <i className={`fa-solid ${isExpanded ? 'fa-video-slash' : 'fa-play'}`}></i>
                    </button>
                    
                    {isExpanded && (
                      <div className="inline-video-drawer animate-slide-down">
                        <div className="video-wrap">
                          <video
                            src={getBackendUrl(`/api/video/${encodeURIComponent(word)}`)}
                            autoPlay
                            loop
                            muted
                            controls
                            playsInline
                          />
                        </div>
                        <div className="video-drawer-info">
                          <p>Watch and mimic the movement trajectory. Drop your wrists when finished to record.</p>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <p className="empty-category-text">No matching signs found.</p>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

export default App;
