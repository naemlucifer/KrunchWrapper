import { useState } from 'react';
import { MessageExtra } from '../utils/types';
import toast from 'react-hot-toast';
import { useAppContext } from '../utils/app.context';

// This file handles uploading extra context items (a.k.a files)
// It only allows text-based files including:
// - Source code files (.js, .py, .java, .cpp, .c, .go, .rs, etc.)
// - Configuration files (.json, .yaml, .yml, .xml, .toml, .ini, etc.)
// - Documentation files (.txt, .md, .rst, .tex, etc.)
// - Script files (.sh, .ps1, .bat, .cmd, etc.)
// - Web files (.html, .css, .scss, .less, etc.)
// - Data files (.csv, .tsv, .sql, etc.)

// Comprehensive list of allowed text file extensions
function isAllowedTextFile(fileName: string): boolean {
  const ext = fileName.toLowerCase().split('.').pop() || '';
  
  // Programming languages
  const codeExtensions = [
    'js', 'jsx', 'ts', 'tsx', 'mjs', 'cjs', // JavaScript/TypeScript
    'py', 'pyw', 'pyi', 'pyx', // Python
    'java', 'scala', 'kt', 'kts', // JVM languages
    'c', 'cpp', 'cxx', 'cc', 'h', 'hpp', 'hxx', // C/C++
    'cs', 'vb', 'fs', 'fsx', // .NET languages
    'go', 'mod', 'sum', // Go
    'rs', 'toml', // Rust
    'php', 'phtml', // PHP
    'rb', 'rake', 'gemspec', // Ruby
    'swift', // Swift
    'dart', // Dart
    'lua', // Lua
    'pl', 'pm', 'pod', // Perl
    'r', 'rmd', 'rnw', // R
    'jl', // Julia
    'elm', // Elm
    'clj', 'cljs', 'cljc', 'edn', // Clojure
    'hs', 'lhs', // Haskell
    'ml', 'mli', // OCaml
    'ex', 'exs', // Elixir
    'erl', 'hrl', // Erlang
    'nim', 'nims', // Nim
    'zig', // Zig
    'v', // V
    'cr', // Crystal
    'pas', 'pp', 'inc', // Pascal
    'f', 'f90', 'f95', 'f03', 'f08', // Fortran
    'asm', 's', // Assembly
    'sol', // Solidity
    'move', // Move
  ];
  
  // Web technologies
  const webExtensions = [
    'html', 'htm', 'xhtml', 'shtml',
    'css', 'scss', 'sass', 'less', 'styl',
    'vue', 'svelte', 'astro',
    'jsp', 'asp', 'aspx', 'php',
    'twig', 'blade', 'mustache', 'hbs', 'handlebars',
  ];
  
  // Configuration and data files
  const configExtensions = [
    'json', 'jsonc', 'json5',
    'yaml', 'yml',
    'xml', 'xsd', 'xsl', 'xslt',
    'toml', 'ini', 'cfg', 'conf', 'config',
    'env', 'dotenv',
    'properties', 'props',
    'plist',
  ];
  
  // Documentation and markup
  const docExtensions = [
    'txt', 'text',
    'md', 'markdown', 'mdown', 'mkd', 'mkdn',
    'rst', 'rest',
    'tex', 'latex', 'ltx',
    'org',
    'adoc', 'asciidoc',
    'textile',
    'wiki',
    'rtf',
  ];
  
  // Scripts and automation
  const scriptExtensions = [
    'sh', 'bash', 'zsh', 'fish', 'ksh', 'csh', 'tcsh',
    'ps1', 'psm1', 'psd1', // PowerShell
    'bat', 'cmd', // Windows batch
    'vbs', 'vba', // VBScript/VBA
    'applescript', 'scpt',
    'au3', // AutoIt
    'ahk', // AutoHotkey
  ];
  
  // Data and query files
  const dataExtensions = [
    'sql', 'pgsql', 'mysql', 'sqlite',
    'csv', 'tsv', 'psv',
    'graphql', 'gql',
    'sparql',
    'cypher',
    'hql', 'hive',
  ];
  
  // Build and project files
  const buildExtensions = [
    'makefile', 'mk', 'mak',
    'cmake', 'cmakelists',
    'gradle', 'gradlew',
    'maven', 'pom',
    'sbt',
    'cabal',
    'package', 'lock',
    'dockerfile',
    'vagrantfile',
    'rakefile',
    'gemfile',
    'podfile',
    'cartfile',
    'dune-project',
  ];
  
  // Log and output files
  const logExtensions = [
    'log', 'logs', 'out', 'err', 'trace',
  ];
  
  // License and readme files
  const specialExtensions = [
    'license', 'licence', 'copying',
    'readme', 'changelog', 'changes', 'news',
    'authors', 'contributors', 'maintainers',
    'todo', 'fixme', 'issues',
    'gitignore', 'gitattributes', 'gitmodules',
    'editorconfig', 'prettierrc', 'eslintrc',
    'htaccess', 'htpasswd',
    'npmrc', 'nvmrc', 'babelrc', 'browserslistrc',
    'tsconfig', 'jsconfig',
  ];
  
  // Files without extensions that are typically text
  if (!ext && fileName.toLowerCase().match(/^(readme|license|changelog|dockerfile|makefile|rakefile|gemfile|podfile|cartfile|vagrantfile|procfile|cmakelists)$/)) {
    return true;
  }
  
  return [
    ...codeExtensions,
    ...webExtensions,
    ...configExtensions,
    ...docExtensions,
    ...scriptExtensions,
    ...dataExtensions,
    ...buildExtensions,
    ...logExtensions,
    ...specialExtensions,
  ].includes(ext);
}

// Interface describing the API returned by the hook
export interface ChatExtraContextApi {
  items?: MessageExtra[]; // undefined if empty, similar to Message['extra']
  addItems: (items: MessageExtra[]) => void;
  removeItem: (idx: number) => void;
  clearItems: () => void;
  onFileAdded: (files: File[]) => void; // used by "upload" button
}

export function useChatExtraContext(): ChatExtraContextApi {
  const [items, setItems] = useState<MessageExtra[]>([]);

  const addItems = (newItems: MessageExtra[]) => {
    setItems((prev) => [...prev, ...newItems]);
  };

  const removeItem = (idx: number) => {
    setItems((prev) => prev.filter((_, i) => i !== idx));
  };

  const clearItems = () => {
    setItems([]);
  };

  const onFileAdded = async (files: File[]) => {
    try {
      for (const file of files) {
        // Check if file extension is allowed for text-only content
        if (!isAllowedTextFile(file.name)) {
          toast.error(
            `${file.name}: Only text-based files are allowed. Supported types include: code files (.js, .py, .java, etc.), documents (.txt, .md, .json), scripts (.sh, .ps1), config files, and more.`
          );
          continue; // Skip this file but continue with others
        }

        // this limit is only to prevent accidental uploads of huge files
        // it can potentially crashes the browser because we read the file as base64
        if (file.size > 50 * 1024 * 1024) {
          toast.error('File is too large. Maximum size is 50MB for text files.');
          continue;
        }

        // Only process as text files now
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target?.result) {
            const content = event.target.result as string;
            if (!isLikelyNotBinary(content)) {
              toast.error(`${file.name}: File appears to be binary. Please upload only text-based files.`);
              return;
            }
            addItems([
              {
                type: 'textFile',
                name: file.name,
                content,
              },
            ]);
            toast.success(`${file.name}: Text file attached successfully.`);
          }
        };
        reader.readAsText(file);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const errorMessage = `Error processing file: ${message}`;
      toast.error(errorMessage);
    }
  };

  return {
    items: items.length > 0 ? items : undefined,
    addItems,
    removeItem,
    clearItems,
    onFileAdded,
  };
}



// WARN: vibe code below
// This code is a heuristic to determine if a string is likely not binary.
// It is necessary because input file can have various mime types which we don't have time to investigate.
// For example, a python file can be text/plain, application/x-python, etc.
function isLikelyNotBinary(str: string): boolean {
  const options = {
    prefixLength: 1024 * 10, // Check the first 10KB of the string
    suspiciousCharThresholdRatio: 0.15, // Allow up to 15% suspicious chars
    maxAbsoluteNullBytes: 2,
  };

  if (!str) {
    return true; // Empty string is considered "not binary" or trivially text.
  }

  const sampleLength = Math.min(str.length, options.prefixLength);
  if (sampleLength === 0) {
    return true; // Effectively an empty string after considering prefixLength.
  }

  let suspiciousCharCount = 0;
  let nullByteCount = 0;

  for (let i = 0; i < sampleLength; i++) {
    const charCode = str.charCodeAt(i);

    // 1. Check for Unicode Replacement Character (U+FFFD)
    // This is a strong indicator if the string was created from decoding bytes as UTF-8.
    if (charCode === 0xfffd) {
      suspiciousCharCount++;
      continue;
    }

    // 2. Check for Null Bytes (U+0000)
    if (charCode === 0x0000) {
      nullByteCount++;
      // We also count nulls towards the general suspicious character count,
      // as they are less common in typical text files.
      suspiciousCharCount++;
      continue;
    }

    // 3. Check for C0 Control Characters (U+0001 to U+001F)
    // Exclude common text control characters: TAB (9), LF (10), CR (13).
    // We can also be a bit lenient with BEL (7) and BS (8) which sometimes appear in logs.
    if (charCode < 32) {
      if (
        charCode !== 9 && // TAB
        charCode !== 10 && // LF
        charCode !== 13 && // CR
        charCode !== 7 && // BEL (Bell) - sometimes in logs
        charCode !== 8 // BS (Backspace) - less common, but possible
      ) {
        suspiciousCharCount++;
      }
    }
    // Characters from 32 (space) up to 126 (~) are printable ASCII.
    // Characters 127 (DEL) is a control character.
    // Characters >= 128 are extended ASCII / multi-byte Unicode.
    // If they resulted in U+FFFD, we caught it. Otherwise, they are valid
    // (though perhaps unusual) Unicode characters from JS's perspective.
    // The main concern is if those higher characters came from misinterpreting
    // a single-byte encoding as UTF-8, which again, U+FFFD would usually flag.
  }

  // Check absolute null byte count
  if (nullByteCount > options.maxAbsoluteNullBytes) {
    return false; // Too many null bytes is a strong binary indicator
  }

  // Check ratio of suspicious characters
  const ratio = suspiciousCharCount / sampleLength;
  return ratio <= options.suspiciousCharThresholdRatio;
}


