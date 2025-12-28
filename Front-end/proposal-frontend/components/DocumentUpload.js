'use client';
import React, { useState } from 'react';
import axios from 'axios';
import styles from './DocumentUpload.module.css';

// --- Icon Components (to avoid huge SVG code blocks in the main return) ---

const UploadIcon = () => (
  <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
    />
  </svg>
);

const SummaryIcon = () => (
  <svg className={styles.icon} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
);

const EmptyStateIcon = () => (
  <svg className={styles.emptyStateIcon} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={1.5}
      d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
    />
  </svg>
);

const Spinner = () => (
  <svg className={styles.spinner} viewBox="0 0 24 24" fill="none">
    <circle
      className={styles.spinnerCircle}
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className={styles.spinnerPath}
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

// --- Main Component ---

export default function DocumentUpload() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [keywords, setKeywords] = useState([]);
  const [error, setError] = useState('');
  const [documentInfo, setDocumentInfo] = useState(null);
  const [summary, setSummary] = useState('');
  const [evaluation, setEvaluation] = useState(null);
  const [extractedMetadata, setExtractedMetadata] = useState(null);
  const [isSuccess, setIsSuccess] = useState(false); // New state for success animation

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    setIsSuccess(false); // Reset success state
    if (selectedFile) {
      const fileExtension = selectedFile.name.split('.').pop()?.toLowerCase();
      const allowedExtensions = ['pdf', 'docx', 'txt'];

      if (fileExtension && allowedExtensions.includes(fileExtension)) {
        setFile(selectedFile);
        setFileName(selectedFile.name);
        setError('');
      } else {
        alert('Invalid file type. Please select a PDF, DOCX, or TXT file.');
        e.target.value = '';
        setFile(null);
        setFileName('');
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      alert('Please select a file to upload.');
      return;
    }

    setLoading(true);
    setIsSuccess(false); // Important: reset success on new submission
    setError('');
    setKeywords([]);
    setDocumentInfo(null);
    setSummary('');
    setEvaluation(null);
    setExtractedMetadata(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(
        'http://localhost:8000/api/documents/upload/',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setDocumentInfo({
        filename: response.data.filename || fileName,
        upload_date: response.data.upload_date || new Date().toISOString(),
        keyword_count: response.data.keywords?.length || 0,
        id: response.data.id || Math.floor(Math.random() * 1000),
        status: response.data.status || 'PROCESSED',
      });

      setKeywords(response.data.keywords || []);
      setSummary(response.data.summary || '');
      setEvaluation(response.data.evaluation || null);
      setExtractedMetadata(response.data.rfp_metadata || null);
      setIsSuccess(true); // Set success state on successful response

    } catch (err) {
      console.error('Error uploading file:', err);
      setError('Failed to upload and process document. Please try again. Check server status.');
      setIsSuccess(false);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setFileName('');
    setLoading(false);
    setError('');
    setKeywords([]);
    setDocumentInfo(null);
    setSummary('');
    setEvaluation(null);
    setExtractedMetadata(null);
    setIsSuccess(false);
  };

  const renderDecisionBadge = (decision) => {
    const decisionText = {
      ACCEPT: 'ACCEPT ✅',
      REJECT: 'REJECT ❌',
      REVIEW: 'REVIEW ⚠️',
    };

    const styleMap = {
      ACCEPT: { backgroundColor: 'var(--success-bg)', color: 'var(--success-color)' },
      REJECT: { backgroundColor: 'var(--error-bg)', color: 'var(--error-color)' },
      REVIEW: { backgroundColor: 'var(--warning-bg)', color: 'var(--warning-color)' },
    };

    const badgeStyle = styleMap[decision] || styleMap['REVIEW'];
    const text = decisionText[decision] || decisionText['REVIEW'];

    return (
      <span className={styles.decisionBadge} style={badgeStyle}>
        {text}
      </span>
    );
  };

  const formatINR = (value) => {
    if (!value || value <= 0) return 'Unknown';
    // Use the maximumFractionDigits: 0 for clean display as per original logic
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0,
    }).format(value);
  };

  const hasResults = keywords.length > 0 || evaluation || extractedMetadata || summary;

  return (
    <div className={styles.pageBackground}>
      <div className={styles.container}>
        <div className={styles.card}>
          {/* Header */}
          <header className={styles.header}>
            <h1 className={styles.title}>
              RFP Evaluation & Keyword Extractor
            </h1>
            <p className={styles.subtitle}>
              Upload RFP documents to extract key details, get a project summary, and receive an automated fit evaluation.
            </p>
          </header>

          {/* Upload Form */}
          <form onSubmit={handleSubmit} className={styles.form}>
            {/* File upload */}
            <div className={styles.formGroup}>
              <div className={`${styles.uploadArea} ${fileName ? styles.uploaded : ''}`}>
                <label className={styles.uploadLabel}>
                  <div className={styles.uploadContent}>
                    <UploadIcon />
                    {fileName ? (
                      <p className={styles.fileName}>
                        Selected: <span className={styles.fileNameHighlight}>{fileName}</span>
                      </p>
                    ) : (
                      <>
                        <p className={styles.uploadText}>
                          <span className={styles.uploadTextBold}>Click to upload</span> or drag and drop
                        </p>
                        <p className={styles.uploadSubtext}>
                          PDF, DOCX or TXT (MAX. 10MB)
                        </p>
                      </>
                    )}
                  </div>
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.docx,.doc,.txt"
                    onChange={handleFileChange}
                    className={styles.fileInput}
                  />
                </label>
              </div>
            </div>

            <div className={styles.buttonGroup}>
              <button
                type="submit"
                disabled={!file || loading}
                className={styles.submitButton}
              >
                {loading ? (
                  <span className={styles.buttonContent}>
                    <Spinner />
                    Processing...
                  </span>
                ) : (
                  <span className={styles.buttonContent}>
                    Upload & Evaluate
                  </span>
                )}
              </button>

              {(file || hasResults) && (
                <button
                  type="button"
                  onClick={handleReset}
                  className={styles.resetButton}
                  disabled={loading}
                >
                  Reset
                </button>
              )}
            </div>
          </form>

          {/* Error Message */}
          {error && <div className={styles.errorMessage}>{error}</div>}

          {/* Results Container - Conditional Rendering with Fade-in */}
          {hasResults && (
            <div className={`${styles.resultsContainer} ${isSuccess ? styles.resultsShow : ''}`}>

              {/* Document Information */}
              {documentInfo && (
                <section className={styles.resultSection}>
                  <h2 className={styles.sectionTitle}>Document Upload Info</h2>
                  <div className={styles.infoGrid}>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Filename:</span> {documentInfo.filename}</div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Uploaded:</span> {new Date(documentInfo.upload_date).toLocaleString()}</div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Keywords Extracted:</span> {documentInfo.keyword_count}</div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Document ID:</span> #{documentInfo.id}</div>
                    {documentInfo.status && (
                      <div className={styles.infoItem}><span className={styles.infoLabel}>Status:</span> {documentInfo.status}</div>
                    )}
                  </div>
                </section>
              )}

              {/* Auto-detected RFP Details */}
              {extractedMetadata && (
                <section className={styles.resultSection}>
                  <h2 className={styles.sectionTitle}>Auto-detected RFP Details</h2>
                  <div className={styles.infoGrid}>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Budget:</span> <strong>{formatINR(extractedMetadata.budget_in_inr)}</strong></div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Timeline:</span> <strong>{extractedMetadata.timeline_weeks ? `${extractedMetadata.timeline_weeks} weeks` : 'Unknown'}</strong></div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Team Size:</span> <strong>{extractedMetadata.team_size_required || 'Unknown'}</strong></div>
                    <div className={styles.infoItem}><span className={styles.infoLabel}>Confidence:</span> {extractedMetadata.extraction_confidence || 'N/A'}</div>
                  </div>
                  {extractedMetadata.extraction_notes && (
                    <p className={styles.noteText}>
                        <span className={styles.infoLabel}>Notes:</span> {extractedMetadata.extraction_notes}
                    </p>
                  )}
                </section>
              )}

              {/* Evaluation section */}
              {evaluation && (
                <section className={styles.resultSection}>
                  <h3 className={styles.sectionTitle}>RFP Evaluation</h3>
                  <div className={styles.evaluationHeader}>
                    <span className={styles.evaluationLabel}>Decision:</span>
                    {renderDecisionBadge(evaluation.decision)}
                  </div>
                  <div className={styles.evaluationGrid}>
                    <div className={styles.scoreItem}>Technical Fit: <span className={styles.scoreValue}>{evaluation.technical_fit_score}%</span></div>
                    <div className={styles.scoreItem}>Budget Fit: <span className={styles.scoreValue}>{evaluation.budget_fit_score}%</span></div>
                    <div className={styles.scoreItem}>Timeline Fit: <span className={styles.scoreValue}>{evaluation.timeline_fit_score}%</span></div>
                    <div className={styles.scoreItem}>Capacity Fit: <span className={styles.scoreValue}>{evaluation.capacity_fit_score}%</span></div>
                    <div className={`${styles.scoreItem} ${styles.scoreOverall}`}>Overall Fit: <span className={styles.scoreValue}>{evaluation.overall_fit_score}%</span></div>
                  </div>
                  <p className={styles.reasoningText}>
                    <span className={styles.infoLabel}>Reasoning:</span> {evaluation.reasoning}
                  </p>
                </section>
              )}

              {/* Summary Section */}
              {summary && (
                <section className={styles.resultSection}>
                  <h3 className={styles.sectionTitle}>
                    <SummaryIcon />
                    Document Summary
                  </h3>
                  <p className={styles.summaryText}>{summary}</p>
                </section>
              )}

              {/* Keywords Display */}
              {keywords.length > 0 && (
                <section className={styles.resultSection}>
                  <div className={styles.keywordsHeader}>
                    <h2 className={styles.sectionTitle}>Extracted Keywords</h2>
                    <span className={styles.keywordsCountBadge}>
                      {keywords.length} terms
                    </span>
                  </div>
                  <div className={styles.keywordsGrid}>
                    {keywords.map((item, index) => (
                      <div key={index} className={styles.keywordCard}>
                        <span className={styles.keywordNumber}>{index + 1}</span>
                        <span className={styles.keywordText}>{item.keyword}</span>
                        <div className={styles.keywordScore}>
                          <span className={styles.keywordScoreValue}>
                            {(item.relevance_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}

          {/* Empty State */}
          {!loading && !hasResults && !error && (
            <div className={styles.emptyState}>
              <EmptyStateIcon />
              <p className={styles.emptyStateText}>
                Upload an RFP document (PDF, DOCX, or TXT) to get started with automated analysis.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}