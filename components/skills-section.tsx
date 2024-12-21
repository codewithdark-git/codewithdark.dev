'use client'

import { useEffect } from 'react'
import { Card } from "@/components/ui/card"
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
    <div className="grid grid-rows-3 md:grid-cols-4 gap-4">
      {Object.entries(skillsData).map(([skill, data]) => (
        <Card 
          key={skill} 
          className="group overflow-hidden hover:shadow-xl transition-all duration-500 hover:-translate-y-2 relative"
        >
          <div className="p-4">
            <div className="flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-xl group-hover:bg-primary/20 transition-colors duration-300">
                    <SkillIcon name={skill} className="text-emerald-400 w-6 h-6 text-primary group-hover:scale-110 transition-transform duration-300" />
                  </div>
                  <div>
                    <h3 className="text-base font-bold group-hover:text-primary transition-colors duration-300">{skill}</h3>
                    <p className="text-xs text-muted-foreground">
                      {data.count} {data.count === 1 ? 'project' : 'projects'} completed
                    </p>
                  </div>
                </div>
                <span className="text-lg font-bold text-primary opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                  {data.percentage}%
                </span>
              </div>
              <div className="w-full bg-secondary/30 h-3 rounded-full overflow-hidden">
                <div 
                  className="bg-primary h-full transition-all duration-1000 ease-out rounded-full relative"
                  style={{ width: '0%' }}
                  data-percentage={data.percentage}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-primary/50 to-primary"></div>
                </div>
              </div>
            </div>
          </div>
          <div className="absolute inset-0 bg-primary/5 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
        </Card>
      ))}
    </div>
  )
}

