'use client'

import { useEffect } from 'react'
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import SkillIcon from "@/components/skill-icon"

interface SkillData {
  count: number
  percentage: number
  icon: string
}

interface SkillsSectionProps {
  skillsData: Record<string, SkillData>
}

export function SkillsSection({ skillsData }: SkillsSectionProps) {
  useEffect(() => {
    const skillBars = document.querySelectorAll('[data-percentage]');
    skillBars.forEach((bar) => {
      const percentage = bar.getAttribute('data-percentage');
      setTimeout(() => {
        (bar as HTMLElement).style.width = `${percentage}%`;
      }, 300);
    });
  }, []);

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
      {Object.entries(skillsData).map(([skill, data]) => (
        <Card key={skill} className="group hover:shadow-lg transition-all duration-300 overflow-hidden">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <SkillIcon name={skill} className="w-8 h-8 text-primary" />
                <h3 className="text-lg font-semibold group-hover:text-primary transition-colors">{skill}</h3>
              </div>
              <span className="text-muted-foreground">{data.percentage}%</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-2.5 mb-4">
              <div 
                className="bg-primary h-2.5 rounded-full transition-all duration-500 ease-out" 
                style={{ width: '0%' }}
                data-percentage={data.percentage}
              ></div>
            </div>
            <p className="text-sm text-muted-foreground">
              {data.count} project{data.count !== 1 ? 's' : ''} completed
            </p>
          </div>
        </Card>
      ))}
    </div>
  )
}

