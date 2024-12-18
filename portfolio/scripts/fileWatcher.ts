import chokidar from 'chokidar';
import { processMarkdownFile } from './markdownProcessor';
import { updateSkills } from './skillsUpdater';

const contentDir = './content';

const watcher = chokidar.watch(contentDir, {
  ignored: /(^|[\/\\])\../,
  persistent: true
});

watcher
  .on('add', async (path) => {
    console.log(`File ${path} has been added`);
    
    if (path.endsWith('.md')) {
      try {
        const { type, data } = await processMarkdownFile(path);
        if (type === 'project') {
          await updateSkills(data.technologies);
        }
        console.log(`Processed ${type}: ${data.title}`);
      } catch (error) {
        console.error(`Error processing file ${path}:`, error);
      }
    }
  });

console.log(`Watching for file changes on ${contentDir}`);

