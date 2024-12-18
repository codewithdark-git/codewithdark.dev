import { 
  SiPython, 
  SiOpencv, 
  SiHuggingface, 
  SiStreamlit, 
  SiFlask, 
  SiGithub,
  SiJavascript,
  SiReact,
  SiNodedotjs,
  SiMongodb,
  SiPostgresql,
  SiDocker,
  //SiKubernetes,
  //SiGooglecloud,
  //SiTensorflow,
  //SiPytorch,
  //SiScikitlearn,
  //SiPandas,
  //SiNumpy
} from 'react-icons/si';
import { IconType } from 'react-icons';
import { FaCode, FaServer, FaImage, FaCloud } from 'react-icons/fa';

interface SkillIconProps {
  name: string;
  className?: string;
}

const iconMap: Record<string, IconType> = {
  "Python Developer": SiPython,
  "OpenCV": SiOpencv,
  "Hugging Face": SiHuggingface,
  "Streamlit": SiStreamlit,
  "Flask": SiFlask,
  "API Development": FaServer,
  "Web Scripting": FaCode,
  "GitHub": SiGithub,
  "Image Processing": FaImage,
  "JavaScript": SiJavascript,
  "React": SiReact,
  "Node.js": SiNodedotjs,
  "MongoDB": SiMongodb,
  "PostgreSQL": SiPostgresql,
  "Docker": SiDocker,
  //"Kubernetes": SiKubernetes,
  //"Google Cloud": SiGooglecloud,
  //"TensorFlow": SiTensorflow,
  //"PyTorch": SiPytorch,
  //"scikit-learn": SiScikitlearn,
  //"Pandas": SiPandas,
  //"NumPy": SiNumpy
  "AWS": FaCloud,
};

const SkillIcon: React.FC<SkillIconProps> = ({ name, className }) => {
  const Icon = iconMap[name] || FaCode;
  return <Icon className={className} />;
};

export default SkillIcon;

