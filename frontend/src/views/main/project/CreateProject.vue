<template>
  <v-container fluid>
    <v-card class="ma-3 pa-3">
      <v-card-title primary-title>
        <div class="headline primary--text">Create Project</div>
      </v-card-title>
      <v-card-text>
        <ProjectForm :project="newProject"></ProjectForm>
      </v-card-text>
      <v-card-actions>
        <v-spacer></v-spacer>
        <v-btn @click="cancel">Cancel</v-btn>
        <v-btn @click="reset">Reset</v-btn>
        <v-btn @click="submit"> Save </v-btn>
      </v-card-actions>
    </v-card>
  </v-container>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import { Project, ProjectUpdate, ProjectCreate } from '@/api';
import { defaultProject } from '@/interfaces';
import ProjectForm from '@/components/ProjectForm.vue';
import { dispatchCreateProject } from '@/store/project/actions';
import { filterUndefined } from '@/utils';

@Component({ components: { ProjectForm } })
export default class CreateProject extends Vue {
  public newProject: ProjectCreate = defaultProject();
  public valid = false;

  public async mounted() {
    this.reset();
  }

  public reset() {
    this.newProject = defaultProject();
    this.$validator.reset();
  }

  public cancel() {
    this.$router.back();
  }

  public async submit() {
    if (await this.$validator.validateAll()) {
      const filteredProject: ProjectCreate = filterUndefined(this.newProject);
      await dispatchCreateProject(this.$store, filteredProject);
      this.$router.push('/main/projects');
    }
  }
}
</script>
