import fs from 'fs/promises';
import path from 'path';

const dataDir = path.join(process.cwd(), 'data');
const skillsFile = path.join(dataDir, 'skills.json');

export async function updateSkills(technologies: string[]) {
  try {
    let skillsData: Record<string, { count: number; percentage: number }> = {};

    // Ensure the data directory exists
    await fs.mkdir(dataDir, { recursive: true });

    // Check if the file exists, if not, create it with an empty object
    try {
      const fileContent = await fs.readFile(skillsFile, 'utf8');
      skillsData = JSON.parse(fileContent);
    } catch (error) {
      // If the file doesn't exist, create an empty file
      await fs.writeFile(skillsFile, JSON.stringify({}), 'utf8');
    }

    technologies.forEach(tech => {
      if (skillsData[tech]) {
        skillsData[tech].count += 1;
        skillsData[tech].percentage = Math.min(100, skillsData[tech].percentage + 5);
      } else {
        skillsData[tech] = { count: 1, percentage: 20 };
      }
    });

    await fs.writeFile(skillsFile, JSON.stringify(skillsData, null, 2));
    console.log('Skills updated successfully');
  } catch (error) {
    console.error('Error updating skills:', error);
  }
}

