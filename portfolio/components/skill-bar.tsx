"use client"

import { useEffect, useState } from 'react'
import { Progress } from "@/components/ui/progress"

interface SkillBarProps {
  skill: string
  percentage: number
}

const SkillBar: React.FC<SkillBarProps> = ({ skill, percentage }) => {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const timer = setTimeout(() => setProgress(percentage), 100)
    return () => clearTimeout(timer)
  }, [percentage])

  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <span className="text-sm font-medium">{skill}</span>
        <span className="text-sm font-medium">{percentage}%</span>
      </div>
      <Progress value={progress} className="w-full" />
    </div>
  )
}

export default SkillBar

