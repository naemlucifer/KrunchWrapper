/*
 * This file is derived from the llama.cpp project WebUI
 * Original source: https://github.com/ggml-org/llama.cpp
 * License: MIT License
 * Modifications: Adapted for KrunchWrapper compression proxy integration
 */

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.scss';
import App from './App.tsx';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
