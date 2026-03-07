import React, { createContext, useContext, useState, useEffect } from 'react';

const CaptionContext = createContext();

export function CaptionProvider({ children }) {
  const [captions, setCaptions] = useState([]);
  const [captionSize, setCaptionSize] = useState(() => {
    return localStorage.getItem('captionSize') || 'medium';
  });

  const [showEmojis, setShowEmojis] = useState(() => {
    return localStorage.getItem('showEmojis') !== 'false';
  });

  const [showTranslations, setShowTranslations] = useState(() => {
    return localStorage.getItem('showTranslations') === 'true';
  });

  const [captionOpacity, setCaptionOpacity] = useState(() => {
    return parseFloat(localStorage.getItem('captionOpacity')) || 0.85;
  });

  const [autoScroll, setAutoScroll] = useState(() => {
    return localStorage.getItem('autoScroll') !== 'false';
  });

  useEffect(() => {
    localStorage.setItem('captionSize', captionSize);
  }, [captionSize]);

  useEffect(() => {
    localStorage.setItem('showEmojis', showEmojis);
  }, [showEmojis]);

  useEffect(() => {
    localStorage.setItem('showTranslations', showTranslations);
  }, [showTranslations]);

  useEffect(() => {
    localStorage.setItem('captionOpacity', captionOpacity);
  }, [captionOpacity]);

  useEffect(() => {
    localStorage.setItem('autoScroll', autoScroll);
  }, [autoScroll]);

  const addCaption = (caption) => {
    // id fallback comes FIRST — caller-provided id takes precedence
    // (important so updateCaption can find the caption later by its id)
    setCaptions(prev => [...prev, { id: Date.now(), ...caption }]);
  };

  const updateCaption = (id, updates) => {
    setCaptions(prev => prev.map(c => c.id === id ? { ...c, ...updates } : c));
  };

  const clearCaptions = () => {
    setCaptions([]);
  };

  const getCaptionSizeClass = () => {
    switch (captionSize) {
      case 'small':
        return 'text-sm';
      case 'medium':
        return 'text-base';
      case 'large':
        return 'text-lg';
      case 'xl':
        return 'text-xl';
      default:
        return 'text-base';
    }
  };

  return (
    <CaptionContext.Provider
      value={{
        captions,
        addCaption,
        updateCaption,
        clearCaptions,
        captionSize,
        setCaptionSize,
        showEmojis,
        setShowEmojis,
        showTranslations,
        setShowTranslations,
        captionOpacity,
        setCaptionOpacity,
        autoScroll,
        setAutoScroll,
        getCaptionSizeClass,
      }}
    >
      {children}
    </CaptionContext.Provider>
  );
}

export function useCaptions() {
  const context = useContext(CaptionContext);
  if (!context) {
    throw new Error('useCaptions must be used within CaptionProvider');
  }
  return context;
}
