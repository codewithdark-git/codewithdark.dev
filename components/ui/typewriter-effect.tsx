"use client"

import { useEffect, useState } from 'react'

interface TypewriterEffectProps {
  words: { text: string }[]
  className?: string
}

export const TypewriterEffect: React.FC<TypewriterEffectProps> = ({ words, className }) => {
  const [currentWordIndex, setCurrentWordIndex] = useState(0)
  const [currentText, setCurrentText] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)

  useEffect(() => {
    const typingInterval = 100 // Adjust for typing speed
    const deletingInterval = 50 // Adjust for deleting speed
    const pauseInterval = 1000 // Pause between words

    const typeWriter = () => {
      const currentWord = words[currentWordIndex].text
      
      if (!isDeleting) {
        setCurrentText(currentWord.substring(0, currentText.length + 1))
        
        if (currentText === currentWord) {
          setIsDeleting(true)
          setTimeout(typeWriter, pauseInterval)
          return
        }
      } else {
        setCurrentText(currentWord.substring(0, currentText.length - 1))
        
        if (currentText === '') {
          setIsDeleting(false)
          setCurrentWordIndex((prevIndex) => (prevIndex + 1) % words.length)
        }
      }

      setTimeout(typeWriter, isDeleting ? deletingInterval : typingInterval)
    }

    typeWriter()

    // Cleanup function
    return () => {
      clearTimeout(typeWriter as unknown as number)
    }
  }, [currentText, currentWordIndex, isDeleting, words])

  return <span className={className}>{currentText}</span>
}
